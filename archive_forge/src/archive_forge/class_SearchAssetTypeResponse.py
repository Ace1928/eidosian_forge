from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchAssetTypeResponse(_messages.Message):
    """Response message for AssetTypesService.Search.

  Fields:
    facets: Returned facets from the search results, showing the aggregated
      buckets.
    items: Returned search results.
    nextPageToken: The next-page continuation token.
  """
    facets = _messages.MessageField('Facet', 1, repeated=True)
    items = _messages.MessageField('SearchResultItem', 2, repeated=True)
    nextPageToken = _messages.StringField(3)