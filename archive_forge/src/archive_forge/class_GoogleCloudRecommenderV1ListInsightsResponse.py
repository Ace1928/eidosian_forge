from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1ListInsightsResponse(_messages.Message):
    """Response to the `ListInsights` method.

  Fields:
    insights: The set of insights for the `parent` resource.
    nextPageToken: A token that can be used to request the next page of
      results. This field is empty if there are no additional results.
  """
    insights = _messages.MessageField('GoogleCloudRecommenderV1Insight', 1, repeated=True)
    nextPageToken = _messages.StringField(2)