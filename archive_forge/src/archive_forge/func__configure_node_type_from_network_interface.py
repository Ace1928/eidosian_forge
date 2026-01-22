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
def _configure_node_type_from_network_interface(node_type: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copies all network interface subnet and security group IDs up to the
    parent node config for the given node type.

    Args:
        node_type (Dict[str, Any]): node type config to bootstrap
    Returns:
        node_type (Dict[str, Any]): The input config with all network interface
        subnet and security group IDs copied into the node config of the
        given node type. If no network interfaces are found, then the
        config is returned unchanged.
    Raises:
        ValueError: If [1] subnet and security group IDs exist at both the
        node config and network interface levels, [2] any network interface
        doesn't have a subnet defined, or [3] any network interface doesn't
        have a security group defined.
    """
    node_type = copy.deepcopy(node_type)
    node_cfg = node_type['node_config']
    if 'NetworkInterfaces' in node_cfg:
        node_type['node_config'] = _configure_subnets_and_groups_from_network_interfaces(node_cfg)
    return node_type