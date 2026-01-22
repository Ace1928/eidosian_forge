from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsListRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of oauth clients to return. If
      unspecified, at most 50 oauth clients will be returned. The maximum
      value is 100; values above 100 are truncated to 100.
    pageToken: Optional. A page token, received from a previous
      `ListOauthClients` call. Provide this to retrieve the subsequent page.
    parent: Required. The parent to list oauth clients for.
    showDeleted: Optional. Whether to return soft-deleted oauth clients.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)