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
def _get_head_node_config_hash(self, config_type: str) -> str:
    hash_key_value = '-'.join([CLOUDWATCH_CONFIG_HASH_TAG_BASE, config_type])
    filters = copy.deepcopy(self._get_current_cluster_session_nodes(self.cluster_name))
    filters.append({'Name': 'tag:{}'.format(TAG_RAY_NODE_KIND), 'Values': [NODE_KIND_HEAD]})
    try:
        instance = list(self.ec2_resource.instances.filter(Filters=filters))
        assert len(instance) == 1, 'More than 1 head node found!'
        for tag in instance[0].tags:
            if tag['Key'] == hash_key_value:
                return tag['Value']
    except botocore.exceptions.ClientError as e:
        logger.warning('{} Error caught when getting value of {} tag on head node'.format(e.response['Error'], hash_key_value))