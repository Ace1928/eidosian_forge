from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderOrganizationsLocationsRecommendersRecommendationsGetRequest(_messages.Message):
    """A RecommenderOrganizationsLocationsRecommendersRecommendationsGetRequest
  object.

  Fields:
    name: Required. Name of the recommendation.
  """
    name = _messages.StringField(1, required=True)