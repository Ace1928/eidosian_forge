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
def _configure_node_cfg_from_launch_template(config: Dict[str, Any], node_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges any launch template data referenced by the given node type's
    node config into the parent node config. Any parameters specified in
    node config override the same parameters in the launch template.

    Note that this merge is simply a bidirectional dictionary update, from
    the node config to the launch template data, and from the launch
    template data to the node config. Thus, the final result captures the
    relative complement of launch template data with respect to node config,
    and allows all subsequent config bootstrapping code paths to act as
    if the complement was explicitly specified in the user's node config. A
    deep merge of nested elements like tag specifications isn't required
    here, since the AWSNodeProvider's ec2.create_instances call will do this
    for us after it fetches the referenced launch template data.

    Args:
        config (Dict[str, Any]): config to bootstrap
        node_cfg (Dict[str, Any]): node config to bootstrap
    Returns:
        node_cfg (Dict[str, Any]): The input node config merged with all launch
        template data. If no launch template data is found, then the node
        config is returned unchanged.
    Raises:
        ValueError: If no launch template is found for the given launch
        template [name|id] and version, or more than one launch template is
        found.
    """
    node_cfg = copy.deepcopy(node_cfg)
    ec2 = _client('ec2', config)
    kwargs = copy.deepcopy(node_cfg['LaunchTemplate'])
    template_version = str(kwargs.pop('Version', '$Default'))
    node_cfg['LaunchTemplate']['Version'] = template_version
    kwargs['Versions'] = [template_version] if template_version else []
    template = ec2.describe_launch_template_versions(**kwargs)
    lt_versions = template['LaunchTemplateVersions']
    if len(lt_versions) != 1:
        raise ValueError(f'Expected to find 1 launch template but found {len(lt_versions)}')
    lt_data = template['LaunchTemplateVersions'][0]['LaunchTemplateData']
    lt_data.update(node_cfg)
    node_cfg.update(lt_data)
    return node_cfg