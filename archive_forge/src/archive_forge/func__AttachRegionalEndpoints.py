from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AttachRegionalEndpoints(self, neg_ref, endpoints):
    """Attaches network endpoints to a regional network endpoint group."""
    request_class = self.messages.ComputeRegionNetworkEndpointGroupsAttachNetworkEndpointsRequest
    nested_request_class = self.messages.RegionNetworkEndpointGroupsAttachEndpointsRequest
    request = request_class(networkEndpointGroup=neg_ref.Name(), project=neg_ref.project, region=neg_ref.region, regionNetworkEndpointGroupsAttachEndpointsRequest=nested_request_class(networkEndpoints=self._GetEndpointMessageList(endpoints)))
    return self._region_service.AttachNetworkEndpoints(request)