from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def AddBgpPeer(self, router_ref, args):
    """Mutate the router so to add a BGP peer."""
    get_router_req = self._messages.EdgenetworkProjectsLocationsZonesRoutersGetRequest(name=router_ref.RelativeName())
    router_object = self._service.Get(get_router_req)
    new_router_object = self.ModifyToAddBgpPeer(args, router_object)
    update_router_req = self._messages.EdgenetworkProjectsLocationsZonesRoutersPatchRequest(name=router_ref.RelativeName(), router=new_router_object, updateMask=self.FIELD_PATH_BGP_PEER)
    return self._service.Patch(update_router_req)