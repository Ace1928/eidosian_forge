from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
def GetExistingPeering(peering_ref):
    """Fetch existing AD domain peering."""
    client = util.GetClientForResource(peering_ref)
    messages = util.GetMessagesForResource(peering_ref)
    get_req = messages.ManagedidentitiesProjectsLocationsGlobalPeeringsGetRequest(name=peering_ref.RelativeName())
    return client.projects_locations_global_peerings.Get(get_req)