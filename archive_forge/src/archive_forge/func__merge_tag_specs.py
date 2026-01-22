import copy
import logging
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List
import botocore
from boto3.resources.base import ServiceResource
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.config import bootstrap_aws
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_CREATE_MAX_RETRIES, BOTO_MAX_RETRIES
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
@staticmethod
def _merge_tag_specs(tag_specs: List[Dict[str, Any]], user_tag_specs: List[Dict[str, Any]]) -> None:
    """
        Merges user-provided node config tag specifications into a base
        list of node provider tag specifications. The base list of
        node provider tag specs is modified in-place.

        This allows users to add tags and override values of existing
        tags with their own, and only applies to the resource type
        "instance". All other resource types are appended to the list of
        tag specs.

        Args:
            tag_specs (List[Dict[str, Any]]): base node provider tag specs
            user_tag_specs (List[Dict[str, Any]]): user's node config tag specs
        """
    for user_tag_spec in user_tag_specs:
        if user_tag_spec['ResourceType'] == 'instance':
            for user_tag in user_tag_spec['Tags']:
                exists = False
                for tag in tag_specs[0]['Tags']:
                    if user_tag['Key'] == tag['Key']:
                        exists = True
                        tag['Value'] = user_tag['Value']
                        break
                if not exists:
                    tag_specs[0]['Tags'] += [user_tag]
        else:
            tag_specs += [user_tag_spec]