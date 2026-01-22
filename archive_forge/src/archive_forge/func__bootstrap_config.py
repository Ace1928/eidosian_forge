import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def _bootstrap_config(config: Dict[str, Any], no_config_cache: bool=False) -> Dict[str, Any]:
    config = prepare_config(config)
    hasher = hashlib.sha1()
    hasher.update(json.dumps([config], sort_keys=True).encode('utf-8'))
    cache_key = os.path.join(tempfile.gettempdir(), 'ray-config-{}'.format(hasher.hexdigest()))
    if os.path.exists(cache_key) and (not no_config_cache):
        config_cache = json.loads(open(cache_key).read())
        if config_cache.get('_version', -1) == CONFIG_CACHE_VERSION:
            try_reload_log_state(config_cache['config']['provider'], config_cache.get('provider_log_info'))
            if log_once('_printed_cached_config_warning'):
                cli_logger.verbose_warning('Loaded cached provider configuration from ' + cf.bold('{}'), cache_key)
                if cli_logger.verbosity == 0:
                    cli_logger.warning('Loaded cached provider configuration')
                cli_logger.warning('If you experience issues with the cloud provider, try re-running the command with {}.', cf.bold('--no-config-cache'))
            return config_cache['config']
        else:
            cli_logger.warning('Found cached cluster config but the version ' + cf.bold('{}') + ' (expected ' + cf.bold('{}') + ') does not match.\nThis is normal if cluster launcher was updated.\nConfig will be re-resolved.', config_cache.get('_version', 'none'), CONFIG_CACHE_VERSION)
    importer = _NODE_PROVIDERS.get(config['provider']['type'])
    if not importer:
        raise NotImplementedError('Unsupported provider {}'.format(config['provider']))
    provider_cls = importer(config['provider'])
    cli_logger.print('Checking {} environment settings', _PROVIDER_PRETTY_NAMES.get(config['provider']['type']))
    try:
        config = provider_cls.fillout_available_node_types_resources(config)
    except Exception as exc:
        if cli_logger.verbosity > 2:
            logger.exception('Failed to autodetect node resources.')
        else:
            cli_logger.warning(f'Failed to autodetect node resources: {str(exc)}. You can see full stack trace with higher verbosity.')
    try:
        validate_config(config)
    except (ModuleNotFoundError, ImportError):
        cli_logger.abort('Not all Ray autoscaler dependencies were found. In Ray 1.4+, the Ray CLI, autoscaler, and dashboard will only be usable via `pip install "ray[default]"`. Please update your install command.')
    resolved_config = provider_cls.bootstrap_config(config)
    if not no_config_cache:
        with open(cache_key, 'w') as f:
            config_cache = {'_version': CONFIG_CACHE_VERSION, 'provider_log_info': try_get_log_state(resolved_config['provider']), 'config': resolved_config}
            f.write(json.dumps(config_cache))
    return resolved_config