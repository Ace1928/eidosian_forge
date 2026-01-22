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
def _usable_subnet_ids(user_specified_subnets: Optional[List[Any]], all_subnets: List[Any], azs: Optional[str], vpc_id_of_sg: Optional[str], use_internal_ips: bool, node_type_key: str) -> Tuple[List[str], str]:
    """Prunes subnets down to those that meet the following criteria.

    Subnets must be:
    * 'Available' according to AWS.
    * Public, unless `use_internal_ips` is specified.
    * In one of the AZs, if AZs are provided.
    * In the given VPC, if a VPC is specified for Security Groups.

    Returns:
        List[str]: Subnets that are usable.
        str: VPC ID of the first subnet.
    """

    def _are_user_subnets_pruned(current_subnets: List[Any]) -> bool:
        return user_specified_subnets is not None and len(current_subnets) != len(user_specified_subnets)

    def _get_pruned_subnets(current_subnets: List[Any]) -> Set[str]:
        current_subnet_ids = {s.subnet_id for s in current_subnets}
        user_specified_subnet_ids = {s.subnet_id for s in user_specified_subnets}
        return user_specified_subnet_ids - current_subnet_ids
    try:
        candidate_subnets = user_specified_subnets if user_specified_subnets is not None else all_subnets
        if vpc_id_of_sg:
            candidate_subnets = [s for s in candidate_subnets if s.vpc_id == vpc_id_of_sg]
        subnets = sorted((s for s in candidate_subnets if s.state == 'available' and (use_internal_ips or s.map_public_ip_on_launch)), reverse=True, key=lambda subnet: subnet.availability_zone)
    except botocore.exceptions.ClientError as exc:
        handle_boto_error(exc, 'Failed to fetch available subnets from AWS.')
        raise exc
    if not subnets:
        cli_logger.abort(f'No usable subnets found for node type {node_type_key}, try manually creating an instance in your specified region to populate the list of subnets and trying this again.\nNote that the subnet must map public IPs on instance launch unless you set `use_internal_ips: true` in the `provider` config.')
    elif _are_user_subnets_pruned(subnets):
        cli_logger.abort(f'The specified subnets for node type {node_type_key} are not usable: {_get_pruned_subnets(subnets)}')
    if azs is not None:
        azs = [az.strip() for az in azs.split(',')]
        subnets = [s for az in azs for s in subnets if s.availability_zone == az]
        if not subnets:
            cli_logger.abort(f'No usable subnets matching availability zone {azs} found for node type {node_type_key}.\nChoose a different availability zone or try manually creating an instance in your specified region to populate the list of subnets and trying this again.')
        elif _are_user_subnets_pruned(subnets):
            cli_logger.abort(f'MISMATCH between specified subnets and Availability Zones! The following Availability Zones were specified in the `provider section`: {azs}.\n The following subnets for node type `{node_type_key}` have no matching availability zone: {list(_get_pruned_subnets(subnets))}.')
    first_subnet_vpc_id = subnets[0].vpc_id
    subnets = [s.subnet_id for s in subnets if s.vpc_id == subnets[0].vpc_id]
    if _are_user_subnets_pruned(subnets):
        subnet_vpcs = {s.subnet_id: s.vpc_id for s in user_specified_subnets}
        cli_logger.abort(f'Subnets specified in more than one VPC for node type `{node_type_key}`! Please ensure that all subnets share the same VPC and retry your request. Subnet VPCs: {{}}', subnet_vpcs)
    return (subnets, first_subnet_vpc_id)