from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSnapshotsGetRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSnapshotsGetRequest object.

  Fields:
    name: Required. The snapshot resource name, in the format `projects/{proje
      ct_id}/locations/{location}/instances/{instance_id}/snapshots/{snapshot_
      id}`
  """
    name = _messages.StringField(1, required=True)