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
def warn_about_bad_start_command(start_commands: List[str], no_monitor_on_head: bool=False) -> None:
    ray_start_cmd = list(filter(lambda x: 'ray start' in x, start_commands))
    if len(ray_start_cmd) == 0:
        cli_logger.warning('Ray runtime will not be started because `{}` is not in `{}`.', cf.bold('ray start'), cf.bold('head_start_ray_commands'))
    autoscaling_config_in_ray_start_cmd = any(('autoscaling-config' in x for x in ray_start_cmd))
    if not (autoscaling_config_in_ray_start_cmd or no_monitor_on_head):
        cli_logger.warning('The head node will not launch any workers because `{}` does not have `{}` set.\nPotential fix: add `{}` to the `{}` command under `{}`.', cf.bold('ray start'), cf.bold('--autoscaling-config'), cf.bold('--autoscaling-config=~/ray_bootstrap_config.yaml'), cf.bold('ray start'), cf.bold('head_start_ray_commands'))