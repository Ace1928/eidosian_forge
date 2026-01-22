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
class SnapshotsAdapter(object):
    """Adapter for the Cloud NetApp Files API Snapshot resource."""

    def __init__(self):
        self.release_track = base.ReleaseTrack.GA
        self.client = netapp_api_util.GetClientInstance(release_track=self.release_track)
        self.messages = netapp_api_util.GetMessagesModule(release_track=self.release_track)

    def UpdateSnapshot(self, snapshot_ref, snapshot_config, update_mask):
        """Send a Patch request for the Cloud NetApp Snapshot."""
        update_request = self.messages.NetappProjectsLocationsVolumesSnapshotsPatchRequest(snapshot=snapshot_config, name=snapshot_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_volumes_snapshots.Patch(update_request)
        return update_op

    def ParseUpdatedSnapshotConfig(self, snapshot_config, description=None, labels=None):
        """Parse update information into an updated Snapshot message."""
        if description is not None:
            snapshot_config.description = description
        if labels is not None:
            snapshot_config.labels = labels
        return snapshot_config