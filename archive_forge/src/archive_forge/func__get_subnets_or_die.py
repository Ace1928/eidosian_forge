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
@lru_cache()
def _get_subnets_or_die(ec2, subnet_ids: Tuple[str]):
    subnet_ids = tuple(Counter(subnet_ids).keys())
    subnets = list(ec2.subnets.filter(Filters=[{'Name': 'subnet-id', 'Values': list(subnet_ids)}]))
    cli_logger.doassert(len(subnets) == len(subnet_ids), 'Not all subnet IDs found: {}', subnet_ids)
    assert len(subnets) == len(subnet_ids), 'Subnet ID not found: {}'.format(subnet_ids)
    return subnets