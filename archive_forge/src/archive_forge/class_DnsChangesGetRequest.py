from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsChangesGetRequest(_messages.Message):
    """A DnsChangesGetRequest object.

  Fields:
    changeId: The identifier of the requested change, from a previous
      ResourceRecordSetsChangeResponse.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or id.
    project: Identifies the project addressed by this request.
  """
    changeId = _messages.StringField(1, required=True)
    managedZone = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)