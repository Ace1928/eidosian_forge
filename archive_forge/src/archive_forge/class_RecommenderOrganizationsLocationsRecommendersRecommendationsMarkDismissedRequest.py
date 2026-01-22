from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderOrganizationsLocationsRecommendersRecommendationsMarkDismissedRequest(_messages.Message):
    """A RecommenderOrganizationsLocationsRecommendersRecommendationsMarkDismis
  sedRequest object.

  Fields:
    googleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest: A
      GoogleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest
      resource to be passed as the request body.
    name: Name of the recommendation.
  """
    googleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest = _messages.MessageField('GoogleCloudRecommenderV1alpha2MarkRecommendationDismissedRequest', 1)
    name = _messages.StringField(2, required=True)