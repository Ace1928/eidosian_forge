from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsPatchRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsPatchRequest object.

  Fields:
    name: Identifier. The resource name of the Replication. Format: `projects/
      {project_id}/locations/{location}/volumes/{volume_id}/replications/{repl
      ication_id}`.
    replication: A Replication resource to be passed as the request body.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
  """
    name = _messages.StringField(1, required=True)
    replication = _messages.MessageField('Replication', 2)
    updateMask = _messages.StringField(3)