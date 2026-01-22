import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def _sha1_hash_file(self, config_type: str) -> str:
    """calculate the config file sha1 hash"""
    config = self.CLOUDWATCH_CONFIG_TYPE_TO_CONFIG_VARIABLE_REPLACE_FUNC.get(config_type)(config_type)
    value = json.dumps(config)
    sha1_res = self._sha1_hash_json(value)
    return sha1_res