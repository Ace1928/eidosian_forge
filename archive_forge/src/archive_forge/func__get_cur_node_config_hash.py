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
def _get_cur_node_config_hash(self, config_type: str) -> str:
    hash_key_value = '-'.join([CLOUDWATCH_CONFIG_HASH_TAG_BASE, config_type])
    try:
        response = self.ec2_client.describe_instances(InstanceIds=[self.node_id])
        reservations = response['Reservations']
        message = 'More than 1 response received from describing current node'
        assert len(reservations) == 1, message
        instances = reservations[0]['Instances']
        assert len(reservations) == 1, message
        tags = instances[0]['Tags']
        hash_value = self._get_default_empty_config_file_hash()
        for tag in tags:
            if tag['Key'] == hash_key_value:
                logger.info('Successfully get cloudwatch {} hash tag value from node {}'.format(config_type, self.node_id))
                hash_value = tag['Value']
        return hash_value
    except botocore.exceptions.ClientError as e:
        logger.warning('{} Error caught when getting hash tag {} tag'.format(e.response['Error'], hash_key_value))