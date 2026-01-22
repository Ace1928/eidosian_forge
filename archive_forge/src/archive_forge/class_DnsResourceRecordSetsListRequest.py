from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsResourceRecordSetsListRequest(_messages.Message):
    """A DnsResourceRecordSetsListRequest object.

  Fields:
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or id.
    maxResults: Optional. Maximum number of results to be returned. If
      unspecified, the server will decide how many results to return.
    name: Restricts the list to return only records with this fully qualified
      domain name.
    pageToken: Optional. A tag returned by a previous list request that was
      truncated. Use this parameter to continue a previous list request.
    project: Identifies the project addressed by this request.
    type: Restricts the list to return only records of this type. If present,
      the "name" parameter must also be present.
  """
    managedZone = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    name = _messages.StringField(3)
    pageToken = _messages.StringField(4)
    project = _messages.StringField(5, required=True)
    type = _messages.StringField(6)