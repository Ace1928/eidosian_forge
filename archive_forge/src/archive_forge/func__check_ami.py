import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def _check_ami(config):
    """Provide helpful message for missing ImageId for node configuration."""
    ami_src_info = {key: 'config' for key in config['available_node_types']}
    _set_config_info(ami_src=ami_src_info)
    region = config['provider']['region']
    default_ami = DEFAULT_AMI.get(region)
    for key, node_type in config['available_node_types'].items():
        node_config = node_type['node_config']
        node_ami = node_config.get('ImageId', '').lower()
        if node_ami in ['', 'latest_dlami']:
            if not default_ami:
                cli_logger.abort(f'Node type `{key}` has no ImageId in its node_config and no default AMI is available for the region `{region}`. ImageId will need to be set manually in your cluster config.')
            else:
                node_config['ImageId'] = default_ami
                ami_src_info[key] = 'dlami'