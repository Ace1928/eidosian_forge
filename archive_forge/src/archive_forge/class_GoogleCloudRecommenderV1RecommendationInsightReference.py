from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1RecommendationInsightReference(_messages.Message):
    """Reference to an associated insight.

  Fields:
    insight: Insight resource name, e.g. projects/[PROJECT_NUMBER]/locations/[
      LOCATION]/insightTypes/[INSIGHT_TYPE_ID]/insights/[INSIGHT_ID]
  """
    insight = _messages.StringField(1)