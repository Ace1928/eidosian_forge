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
def _put_cloudwatch_alarm(self) -> None:
    """put CloudWatch metric alarms read from config"""
    param_name = self._get_ssm_param_name(CloudwatchConfigType.ALARM.value)
    data = json.loads(self._get_ssm_param(param_name))
    for item in data:
        item_out = copy.deepcopy(item)
        self._replace_all_config_variables(item_out, self.node_id, self.cluster_name, self.provider_config['region'])
        self.cloudwatch_client.put_metric_alarm(**item_out)
    logger.info('Successfully put alarms to CloudWatch console')