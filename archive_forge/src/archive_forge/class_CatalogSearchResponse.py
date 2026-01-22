from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CatalogSearchResponse(_messages.Message):
    """Response message for CatalogsService.SearchCatalog.

  Fields:
    items: Returned search results.
    nextPageToken: The next-page continuation token.
  """
    items = _messages.MessageField('SearchResultItem', 1, repeated=True)
    nextPageToken = _messages.StringField(2)