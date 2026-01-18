import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def translate_trivial_legacy_config(config: Dict[str, Any]):
    """
    Drop empty deprecated fields ("head_node" and "worker_node").
    """
    REMOVABLE_FIELDS = ['head_node', 'worker_nodes']
    for field in REMOVABLE_FIELDS:
        if field in config and (not config[field]):
            logger.warning(f'Dropping the empty legacy field {field}. {field}is not supported for ray>=2.0.0. It is recommended to remove{field} from the cluster config.')
            del config[field]