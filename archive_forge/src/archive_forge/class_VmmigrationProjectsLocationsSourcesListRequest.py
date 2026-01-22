from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesListRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesListRequest object.

  Fields:
    filter: Optional. The filter request.
    orderBy: Optional. the order by fields for the result.
    pageSize: Optional. The maximum number of sources to return. The service
      may return fewer than this value. If unspecified, at most 500 sources
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Required. A page token, received from a previous `ListSources`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListSources` must match the call that
      provided the page token.
    parent: Required. The parent, which owns this collection of sources.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)