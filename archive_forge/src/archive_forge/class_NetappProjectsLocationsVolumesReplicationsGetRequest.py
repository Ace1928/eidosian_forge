from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsGetRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsGetRequest object.

  Fields:
    name: Required. The replication resource name, in the format `projects/{pr
      oject_id}/locations/{location}/volumes/{volume_id}/replications/{replica
      tion_id}`
  """
    name = _messages.StringField(1, required=True)