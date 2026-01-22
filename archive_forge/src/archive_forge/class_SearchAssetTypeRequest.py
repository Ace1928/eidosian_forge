from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchAssetTypeRequest(_messages.Message):
    """Request message for AssetTypesService.Search.

  Enums:
    TypeValueValuesEnum: By default, search at segment level.

  Fields:
    facetSelections: Facets to be selected.
    pageSize: The maximum number of items to return. If unspecified, server
      will pick an appropriate default. Server may return fewer items than
      requested. A caller should only rely on response's next_page_token to
      determine if there are more realms left to be queried.
    pageToken: The next_page_token value returned from a previous Search
      request, if any.
    query: Search query.
    type: By default, search at segment level.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """By default, search at segment level.

    Values:
      SEARCH_REQUEST_TYPE_UNSPECIFIED: Unspecified type.
      SEARCH_REQUEST_TYPE_ASSET: Video-level search. That is, search over
        videos and video-level metadata.
      SEARCH_REQUEST_TYPE_SEGMENT: Segment-level search. That is, search over
        segments within videos and annotations.
    """
        SEARCH_REQUEST_TYPE_UNSPECIFIED = 0
        SEARCH_REQUEST_TYPE_ASSET = 1
        SEARCH_REQUEST_TYPE_SEGMENT = 2
    facetSelections = _messages.MessageField('Facet', 1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    query = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)