from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def RestoreVolumeSnapshot(self, request, global_params=None):
    """Uses the specified snapshot to restore its parent volume. Returns INVALID_ARGUMENT if called for a non-boot volume.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RestoreVolumeSnapshot')
    return self._RunMethod(config, request, global_params=global_params)