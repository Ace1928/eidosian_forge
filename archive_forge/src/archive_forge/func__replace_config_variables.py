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
def _replace_config_variables(self, string: str, node_id: str, cluster_name: str, region: str) -> str:
    """
        replace known config variable occurrences in the input string
        does not replace variables with undefined or empty strings
        """
    if node_id:
        string = string.replace('{instance_id}', node_id)
    if cluster_name:
        string = string.replace('{cluster_name}', cluster_name)
    if region:
        string = string.replace('{region}', region)
    return string