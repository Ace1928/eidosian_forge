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
def _replace_alarm_config_vars(self, config_type: str) -> List[str]:
    """
        replace {instance_id}, {region}, {cluster_name}
        variable occurrences in cloudwatch alarm config file
        """
    data = self._load_config_file(config_type)
    param_data = []
    for item in data:
        item_out = copy.deepcopy(item)
        self._replace_all_config_variables(item_out, self.node_id, self.cluster_name, self.provider_config['region'])
        param_data.append(item_out)
    return param_data