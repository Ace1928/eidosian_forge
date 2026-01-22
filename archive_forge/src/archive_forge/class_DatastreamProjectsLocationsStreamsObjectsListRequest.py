from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsObjectsListRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsObjectsListRequest object.

  Fields:
    pageSize: Maximum number of objects to return. Default is 50. The maximum
      value is 1000; values above 1000 will be coerced to 1000.
    pageToken: Page token received from a previous `ListStreamObjectsRequest`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListStreamObjectsRequest` must match the
      call that provided the page token.
    parent: Required. The parent stream that owns the collection of objects.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)