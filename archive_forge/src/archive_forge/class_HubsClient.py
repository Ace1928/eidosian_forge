from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.calliope import base
class HubsClient(object):
    """Client for hub service in network connectivity API."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        self.release_track = release_track
        self.client = networkconnectivity_util.GetClientInstance(release_track)
        self.messages = networkconnectivity_util.GetMessagesModule(release_track)
        self.hub_service = self.client.projects_locations_global_hubs
        self.operation_service = self.client.projects_locations_operations

    def ListHubSpokes(self, hub_ref, spoke_locations=None, limit=None, filter_expression=None, order_by='', page_size=None, page_token=None, view=None):
        """Call API to list spokes."""
        list_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest(name=hub_ref.RelativeName(), spokeLocations=spoke_locations, filter=filter_expression, orderBy=order_by, pageSize=page_size, pageToken=page_token, view=view)
        return list_pager.YieldFromList(self.hub_service, list_req, field='spokes', limit=limit, batch_size_attribute='pageSize', method='ListSpokes')

    def AcceptSpoke(self, hub_ref, spoke):
        """Call API to accept a spoke into a hub."""
        accept_hub_spoke_req = self.messages.AcceptHubSpokeRequest(spokeUri=spoke)
        accept_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest(name=hub_ref.RelativeName(), acceptHubSpokeRequest=accept_hub_spoke_req)
        return self.hub_service.AcceptSpoke(accept_req)

    def RejectSpoke(self, hub_ref, spoke, details):
        """Call API to reject a spoke from a hub."""
        reject_hub_spoke_req = self.messages.RejectHubSpokeRequest(spokeUri=spoke, details=details)
        reject_req = self.messages.NetworkconnectivityProjectsLocationsGlobalHubsRejectSpokeRequest(name=hub_ref.RelativeName(), rejectHubSpokeRequest=reject_hub_spoke_req)
        return self.hub_service.RejectSpoke(reject_req)