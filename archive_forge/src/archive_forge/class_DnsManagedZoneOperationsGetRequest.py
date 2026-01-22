from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsManagedZoneOperationsGetRequest(_messages.Message):
    """A DnsManagedZoneOperationsGetRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    managedZone: Identifies the managed zone addressed by this request.
    operation: Identifies the operation addressed by this request (ID of the
      operation).
    project: Identifies the project addressed by this request.
  """
    clientOperationId = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    operation = _messages.StringField(3, required=True)
    project = _messages.StringField(4, required=True)