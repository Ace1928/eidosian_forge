from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpOrganizationsLocationsConnectionsSearchRequest(_messages.Message):
    """A DlpOrganizationsLocationsConnectionsSearchRequest object.

  Fields:
    filter: Optional. * Supported fields/values - `state` -
      MISSING|AVAILABLE|ERROR
    pageSize: Optional. Number of results per page, max 1000.
    pageToken: Optional. Page token from a previous page to return the next
      set of results. If set, all other request fields must match the original
      request.
    parent: Required. Parent name, typically an organization, without
      location. For example: "organizations/12345678".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)