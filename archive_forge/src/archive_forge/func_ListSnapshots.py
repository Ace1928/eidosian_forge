from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ListSnapshots(self, volume_ref, limit=None):
    """Make API calls to List active Cloud NetApp Volume Snapshots.

    Args:
      volume_ref: The parent Volume to list NetApp Volume Snapshots.
      limit: The number of Cloud NetApp Volume Snapshots to limit the results
        to. This limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp Volume Snapshots.
    """
    request = self.messages.NetappProjectsLocationsVolumesSnapshotsListRequest(parent=volume_ref)
    response = self.client.projects_locations_volumes_snapshots.List(request)
    for location in response.unreachable:
        log.warning('Location {} may be unreachable.'.format(location))
    return list_pager.YieldFromList(self.client.projects_locations_volumes_snapshots, request, field=constants.SNAPSHOT_RESOURCE, limit=limit, batch_size_attribute='pageSize')