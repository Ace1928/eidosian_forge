from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelsListRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelsListRequest object.

  Fields:
    orderBy: The sorting order of the resources returned. Value should be a
      comma-separated list of fields. The default sorting order is ascending.
      To specify descending order for a field, append a `desc` suffix; for
      example: `name desc, channel_id`.
    pageSize: The maximum number of channels to return on each page. Note: The
      service may send fewer.
    pageToken: The page token; provide the value from the `next_page_token`
      field in a previous `ListChannels` call to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListChannels` must
      match the call that provided the page token.
    parent: Required. The parent collection to list channels on.
  """
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)