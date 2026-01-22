from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsTriggersListRequest(_messages.Message):
    """A EventarcProjectsLocationsTriggersListRequest object.

  Fields:
    filter: Filter field. Used to filter the Triggers to be listed. Possible
      filters are described in https://google.aip.dev/160. For example, using
      "?filter=destination:gke" would list only Triggers with a gke
      destination.
    orderBy: The sorting order of the resources returned. Value should be a
      comma-separated list of fields. The default sorting order is ascending.
      To specify descending order for a field, append a `desc` suffix; for
      example: `name desc, trigger_id`.
    pageSize: The maximum number of triggers to return on each page. Note: The
      service may send fewer.
    pageToken: The page token; provide the value from the `next_page_token`
      field in a previous `ListTriggers` call to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListTriggers` must
      match the call that provided the page token.
    parent: Required. The parent collection to list triggers on.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)