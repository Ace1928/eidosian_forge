from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.instance_groups.flags import AutoDeleteFlag
from googlecloudsdk.command_lib.compute.instance_groups.flags import STATEFUL_IP_DEFAULT_INTERFACE_NAME
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_disk_getter
import six
def CallApplyUpdatesToInstances(holder, igm_ref, instances, minimal_action):
    """Calls proper (zonal or reg.) resource for applying updates to instances."""
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        operation_collection = 'compute.zoneOperations'
        service = holder.client.apitools_client.instanceGroupManagers
        minimal_action = holder.client.messages.InstanceGroupManagersApplyUpdatesRequest.MinimalActionValueValuesEnum(minimal_action.upper())
        apply_request = GetApplyUpdatesToInstancesRequestsZonal(holder, igm_ref, instances, minimal_action)
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        operation_collection = 'compute.regionOperations'
        service = holder.client.apitools_client.regionInstanceGroupManagers
        minimal_action = holder.client.messages.RegionInstanceGroupManagersApplyUpdatesRequest.MinimalActionValueValuesEnum(minimal_action.upper())
        apply_request = GetApplyUpdatesToInstancesRequestsRegional(holder, igm_ref, instances, minimal_action)
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    apply_operation = service.ApplyUpdatesToInstances(apply_request)
    apply_operation_ref = holder.resources.Parse(apply_operation.selfLink, collection=operation_collection)
    return apply_operation_ref