from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerOrganizationsGcpUserAccessBindingsListRequest(_messages.Message):
    """A AccesscontextmanagerOrganizationsGcpUserAccessBindingsListRequest
  object.

  Fields:
    pageSize: Optional. Maximum number of items to return. The server may
      return fewer items. If left blank, the server may return any number of
      items.
    pageToken: Optional. If left blank, returns the first page. To enumerate
      all items, use the next_page_token from your previous list operation.
    parent: Required. Example: "organizations/256"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)