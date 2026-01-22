from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsConnectionProfilesListRequest(_messages.Message):
    """A DatastreamProjectsLocationsConnectionProfilesListRequest object.

  Fields:
    filter: Filter request.
    orderBy: Order by fields for the result.
    pageSize: Maximum number of connection profiles to return. If unspecified,
      at most 50 connection profiles will be returned. The maximum value is
      1000; values above 1000 will be coerced to 1000.
    pageToken: Page token received from a previous `ListConnectionProfiles`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListConnectionProfiles` must match the
      call that provided the page token.
    parent: Required. The parent that owns the collection of connection
      profiles.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)