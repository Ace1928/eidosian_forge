from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderFoldersLocationsRecommendersRecommendationsMarkClaimedRequest(_messages.Message):
    """A
  RecommenderFoldersLocationsRecommendersRecommendationsMarkClaimedRequest
  object.

  Fields:
    googleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest: A
      GoogleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest resource
      to be passed as the request body.
    name: Name of the recommendation.
  """
    googleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkRecommendationClaimedRequest', 1)
    name = _messages.StringField(2, required=True)