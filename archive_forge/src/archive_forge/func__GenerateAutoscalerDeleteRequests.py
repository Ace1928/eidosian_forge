from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import text
from six.moves import zip
def _GenerateAutoscalerDeleteRequests(self, holder, project, mig_requests):
    """Generates Delete requestes for autoscalers attached to instance groups.

    Args:
      holder: ComputeApiHolder, object encapsulating compute api.
      project: str, project this request should apply to.
      mig_requests: Messages which will be sent to delete instance group
        managers.

    Returns:
      Messages, which will be sent to delete autoscalers.
    """
    mig_requests = list(zip(*mig_requests))[2] if mig_requests else []
    zone_migs = [(request.instanceGroupManager, 'zone', managed_instance_groups_utils.CreateZoneRef(holder.resources, request)) for request in mig_requests if hasattr(request, 'zone') and request.zone is not None]
    region_migs = [(request.instanceGroupManager, 'region', managed_instance_groups_utils.CreateRegionRef(holder.resources, request)) for request in mig_requests if hasattr(request, 'region') and request.region is not None]
    zones = list(zip(*zone_migs))[2] if zone_migs else []
    regions = list(zip(*region_migs))[2] if region_migs else []
    client = holder.client.apitools_client
    messages = client.MESSAGES_MODULE
    autoscalers_to_delete = managed_instance_groups_utils.AutoscalersForMigs(migs=zone_migs + region_migs, autoscalers=managed_instance_groups_utils.AutoscalersForLocations(zones=zones, regions=regions, client=holder.client))
    requests = []
    for autoscaler in autoscalers_to_delete:
        if autoscaler.zone:
            service = client.autoscalers
            request = messages.ComputeAutoscalersDeleteRequest(zone=path_simplifier.Name(autoscaler.zone))
        else:
            service = client.regionAutoscalers
            request = messages.ComputeRegionAutoscalersDeleteRequest(region=path_simplifier.Name(autoscaler.region))
        request.autoscaler = autoscaler.name
        request.project = project
        requests.append((service, 'Delete', request))
    return requests