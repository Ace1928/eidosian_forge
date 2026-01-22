from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelConnectionsListRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelConnectionsListRequest object.

  Fields:
    pageSize: The maximum number of channel connections to return on each
      page. Note: The service may send fewer responses.
    pageToken: The page token; provide the value from the `next_page_token`
      field in a previous `ListChannelConnections` call to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListChannelConnetions` match the call that provided the page token.
    parent: Required. The parent collection from which to list channel
      connections.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)