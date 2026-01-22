from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderRecommendersListRequest(_messages.Message):
    """A RecommenderRecommendersListRequest object.

  Fields:
    pageSize: Optional. The number of RecommenderTypes to return per page. The
      service may return fewer than this value.
    pageToken: Optional. A page token, received from a previous
      `ListRecommenders` call. Provide this to retrieve the subsequent page.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)