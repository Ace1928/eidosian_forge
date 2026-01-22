from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking import utils
def InitializeZone(self, zone_ref):
    """Initialzie a specified zone."""
    zone_init_req = self._messages.EdgenetworkProjectsLocationsZonesInitializeRequest(name=zone_ref.RelativeName())
    return self._service.Initialize(zone_init_req)