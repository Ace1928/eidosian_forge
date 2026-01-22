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
def _ensure_cwa_installed_ssm(self, node_id: str) -> bool:
    """
        Check if Unified Cloudwatch Agent is installed via ssm run command.
        If not, notify user to use an AMI with
        the Unified CloudWatch Agent installed.
        """
    logger.info('Checking Unified Cloudwatch Agent status on node {}'.format(node_id))
    parameters_status_cwa = {'action': ['status'], 'mode': ['ec2']}
    self._ec2_health_check_waiter(node_id)
    cmd_invocation_res = self._ssm_command_waiter('AmazonCloudWatch-ManageAgent', parameters_status_cwa, node_id, False)
    cwa_installed = cmd_invocation_res.get(node_id, False)
    if not cwa_installed:
        logger.warning('Unified CloudWatch Agent not installed on {}. Ray logs, metrics not picked up. Please use an AMI with Unified CloudWatch Agent installed.'.format(node_id))
        return False
    else:
        return True