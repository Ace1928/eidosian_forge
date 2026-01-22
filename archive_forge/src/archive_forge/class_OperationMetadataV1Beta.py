from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationMetadataV1Beta(_messages.Message):
    """Metadata for the given google.longrunning.Operation.

  Fields:
    createVersionMetadata: A CreateVersionMetadataV1Beta attribute.
    endTime: Time that this operation completed.@OutputOnly
    ephemeralMessage: Ephemeral message that may change every time the
      operation is polled. @OutputOnly
    insertTime: Time that this operation was created.@OutputOnly
    method: API method that initiated this operation. Example:
      google.appengine.v1beta.Versions.CreateVersion.@OutputOnly
    target: Name of the resource that this operation is acting on. Example:
      apps/myapp/services/default.@OutputOnly
    user: User who requested this operation.@OutputOnly
    warning: Durable messages that persist on every operation poll.
      @OutputOnly
  """
    createVersionMetadata = _messages.MessageField('CreateVersionMetadataV1Beta', 1)
    endTime = _messages.StringField(2)
    ephemeralMessage = _messages.StringField(3)
    insertTime = _messages.StringField(4)
    method = _messages.StringField(5)
    target = _messages.StringField(6)
    user = _messages.StringField(7)
    warning = _messages.StringField(8, repeated=True)