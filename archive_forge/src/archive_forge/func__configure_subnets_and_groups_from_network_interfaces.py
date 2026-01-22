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
def _configure_subnets_and_groups_from_network_interfaces(node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copies all network interface subnet and security group IDs into their
    parent node config.

    Args:
        node_cfg (Dict[str, Any]): node config to bootstrap
    Returns:
        node_cfg (Dict[str, Any]): node config with all copied network
        interface subnet and security group IDs
    Raises:
        ValueError: If [1] subnet and security group IDs exist at both the
        node config and network interface levels, [2] any network interface
        doesn't have a subnet defined, or [3] any network interface doesn't
        have a security group defined.
    """
    node_cfg = copy.deepcopy(node_cfg)
    conflict_keys = ['SubnetId', 'SubnetIds', 'SecurityGroupIds']
    if any((conflict in node_cfg for conflict in conflict_keys)):
        raise ValueError('If NetworkInterfaces are defined, subnets and security groups must ONLY be given in each NetworkInterface.')
    subnets = _subnets_in_network_config(node_cfg)
    if not all(subnets):
        raise ValueError('NetworkInterfaces are defined but at least one is missing a subnet. Please ensure all interfaces have a subnet assigned.')
    security_groups = _security_groups_in_network_config(node_cfg)
    if not all(security_groups):
        raise ValueError('NetworkInterfaces are defined but at least one is missing a security group. Please ensure all interfaces have a security group assigned.')
    node_cfg['SubnetIds'] = subnets
    node_cfg['SecurityGroupIds'] = list(itertools.chain(*security_groups))
    return node_cfg