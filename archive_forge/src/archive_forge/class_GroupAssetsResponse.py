from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupAssetsResponse(_messages.Message):
    """Response message for grouping by assets.

  Fields:
    groupByResults: Group results. There exists an element for each existing
      unique combination of property/values. The element contains a count for
      the number of times those specific property/values appear.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
    readTime: Time used for executing the groupBy request.
    totalSize: The total number of results matching the query.
  """
    groupByResults = _messages.MessageField('GroupResult', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    readTime = _messages.StringField(3)
    totalSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)