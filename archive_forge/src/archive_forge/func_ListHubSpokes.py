from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
def ListHubSpokes(self, hub_ref, spoke_locations=None, limit=None, filter_expression=None, order_by='', page_size=None, page_token=None, view=None):
    """Call API to list spokes."""
    list_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest(name=hub_ref.RelativeName(), spokeLocations=spoke_locations, filter=filter_expression, orderBy=order_by, pageSize=page_size, pageToken=page_token, view=view)
    return list_pager.YieldFromList(self.hub_service, list_req, field='spokes', limit=limit, batch_size_attribute='pageSize', method='ListSpokes')