from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed.instance_configs import instance_configs_messages
from googlecloudsdk.core import properties
from six.moves import map
@staticmethod
def _GetDeletePerInstanceConfigRequests(holder, igm_ref, instances):
    """Returns a delete message for instance group manager."""
    messages = holder.client.messages
    req = messages.InstanceGroupManagersDeletePerInstanceConfigsReq(names=Delete._GetInstanceNameListFromUrlList(holder, instances))
    return messages.ComputeInstanceGroupManagersDeletePerInstanceConfigsRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersDeletePerInstanceConfigsReq=req, project=igm_ref.project, zone=igm_ref.zone)