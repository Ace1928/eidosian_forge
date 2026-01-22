from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResourceRecordSetsCreateRequest(_messages.Message):
    """A DnsResourceRecordSetsCreateRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    project: Identifies the project addressed by this request.
    resourceRecordSet: A ResourceRecordSet resource to be passed as the
      request body.
  """
    clientOperationId = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)
    resourceRecordSet = _messages.MessageField('ResourceRecordSet', 4)