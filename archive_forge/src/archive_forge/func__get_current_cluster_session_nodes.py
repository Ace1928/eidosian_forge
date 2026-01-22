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
def _get_current_cluster_session_nodes(self, cluster_name: str) -> List[dict]:
    filters = [{'Name': 'instance-state-name', 'Values': ['pending', 'running']}, {'Name': 'tag:{}'.format(TAG_RAY_CLUSTER_NAME), 'Values': [cluster_name]}]
    return filters