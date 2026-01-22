from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2InsightRecommendationReference(_messages.Message):
    """Reference to an associated recommendation.

  Fields:
    recommendation: Recommendation resource name, e.g. projects/[PROJECT_NUMBE
      R]/locations/[LOCATION]/recommenders/[RECOMMENDER_ID]/recommendations/[R
      ECOMMENDATION_ID]
  """
    recommendation = _messages.StringField(1)