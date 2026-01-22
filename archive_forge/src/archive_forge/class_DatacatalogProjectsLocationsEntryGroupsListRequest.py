from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsListRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of items to return. Default is 10.
      Maximum limit is 1000. Throws an invalid argument if `page_size` is
      greater than 1000.
    pageToken: Optional. Pagination token that specifies the next page to
      return. If empty, returns the first page.
    parent: Required. The name of the location that contains the entry groups
      to list. Can be provided as a URL.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)