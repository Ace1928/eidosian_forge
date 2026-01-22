from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapshotRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapsho
  tRequest object.

  Fields:
    restoreVolumeSnapshotRequest: A RestoreVolumeSnapshotRequest resource to
      be passed as the request body.
    volumeSnapshot: Required. Name of the snapshot which will be used to
      restore its parent volume.
  """
    restoreVolumeSnapshotRequest = _messages.MessageField('RestoreVolumeSnapshotRequest', 1)
    volumeSnapshot = _messages.StringField(2, required=True)