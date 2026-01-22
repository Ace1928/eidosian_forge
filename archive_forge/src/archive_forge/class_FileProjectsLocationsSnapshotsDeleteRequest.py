from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsSnapshotsDeleteRequest(_messages.Message):
    """A FileProjectsLocationsSnapshotsDeleteRequest object.

  Fields:
    name: Required. The snapshot resource name, in the format
      `projects/{project_number}/locations/{location}/snapshots/{snapshot_id}`
  """
    name = _messages.StringField(1, required=True)