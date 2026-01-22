from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils as mig_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_instance_groups_flags
@staticmethod
def _MakePatchRequest(client, igm_ref, igm_updated_resource):
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        service = client.apitools_client.instanceGroupManagers
        request = client.messages.ComputeInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=igm_updated_resource, project=igm_ref.project, zone=igm_ref.zone)
    else:
        service = client.apitools_client.regionInstanceGroupManagers
        request = client.messages.ComputeRegionInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=igm_updated_resource, project=igm_ref.project, region=igm_ref.region)
    return client.MakeRequests([(service, 'Patch', request)])