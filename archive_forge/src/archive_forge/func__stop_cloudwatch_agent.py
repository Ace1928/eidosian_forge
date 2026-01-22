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
def _stop_cloudwatch_agent(self) -> None:
    """stop Unified CloudWatch Agent"""
    logger.info('Stopping Unified CloudWatch Agent package on node {}.'.format(self.node_id))
    parameters_stop_cwa = {'action': ['stop'], 'mode': ['ec2']}
    self._ssm_command_waiter('AmazonCloudWatch-ManageAgent', parameters_stop_cwa, self.node_id, False)
    logger.info('Unified CloudWatch Agent stopped on node {}.'.format(self.node_id))