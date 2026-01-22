from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest
  object.

  Fields:
    name: Required. The resource name of the replication, in the format of pro
      jects/{project_id}/locations/{location}/volumes/{volume_id}/replications
      /{replication_id}.
    reverseReplicationDirectionRequest: A ReverseReplicationDirectionRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    reverseReplicationDirectionRequest = _messages.MessageField('ReverseReplicationDirectionRequest', 2)