from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@staticmethod
def _MakeRmigPatchRequest(client, igm_ref, args):
    service = client.apitools_client.regionInstanceGroupManagers
    request = client.messages.ComputeRegionInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=ResizeAlpha._MakeIgmPatchResource(client, args), project=igm_ref.project, region=igm_ref.region)
    return client.MakeRequests([(service, 'Patch', request)])