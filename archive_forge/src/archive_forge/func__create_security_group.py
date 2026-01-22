import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def _create_security_group(config, vpc_id, group_name):
    client = _client('ec2', config)
    client.create_security_group(Description='Auto-created security group for Ray workers', GroupName=group_name, VpcId=vpc_id)
    security_group = _get_security_group(config, vpc_id, group_name)
    cli_logger.doassert(security_group, 'Failed to create security group')
    cli_logger.verbose('Created new security group {}', cf.bold(security_group.group_name), _tags=dict(id=security_group.id))
    cli_logger.doassert(security_group, 'Failed to create security group')
    assert security_group, 'Failed to create security group'
    return security_group