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
def _upload_config_to_ssm(self, param: Dict[str, Any], config_type: str):
    param_name = self._get_ssm_param_name(config_type)
    self._put_ssm_param(param, param_name)