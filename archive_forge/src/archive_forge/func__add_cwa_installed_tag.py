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
def _add_cwa_installed_tag(self, node_id: str) -> None:
    self.ec2_client.create_tags(Resources=[node_id], Tags=[{'Key': CLOUDWATCH_AGENT_INSTALLED_TAG, 'Value': 'True'}])
    logger.info('Successfully add Unified CloudWatch Agent installed tag on {}'.format(node_id))