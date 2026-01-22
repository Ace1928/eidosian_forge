from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResourceRecordSetsDeleteRequest(_messages.Message):
    """A DnsResourceRecordSetsDeleteRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    name: Fully qualified domain name.
    project: Identifies the project addressed by this request.
    type: RRSet type.
  """
    clientOperationId = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    name = _messages.StringField(3, required=True)
    project = _messages.StringField(4, required=True)
    type = _messages.StringField(5, required=True)