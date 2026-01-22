from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesListRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesListRequest object.

  Fields:
    filter: Supported field for filter is 'service' and value is 'dataplex'.
      Eg: service=dataplex.
    pageSize: The maximum number of items to return. Must be a value between 1
      and 1000 inclusively. If not set, defaults to 50.
    pageToken: The pagination token of the next results page. If not set, the
      first page is returned. The token is returned in the response to a
      previous list request.
    parent: Required. Resource name of the project to list the taxonomies of.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)