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
def _ec2_health_check_waiter(self, node_id: str) -> None:
    try:
        logger.info('Waiting for EC2 instance health checks to complete before configuring Unified Cloudwatch Agent. This may take a few minutes...')
        waiter = self.ec2_client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=[node_id])
    except botocore.exceptions.WaiterError as e:
        logger.error('Failed while waiting for EC2 instance checks to complete: {}'.format(e.message))
        raise e