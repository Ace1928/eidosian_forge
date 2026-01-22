from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsChangesCreateRequest(_messages.Message):
    """A DnsChangesCreateRequest object.

  Fields:
    change: A Change resource to be passed as the request body.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or id.
    project: Identifies the project addressed by this request.
  """
    change = _messages.MessageField('Change', 1)
    managedZone = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)