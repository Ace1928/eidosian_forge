from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityAssessmentResultScoringResultAssessmentRecommendationRecommendationLink(_messages.Message):
    """The format for a link in the recommendation.

  Fields:
    text: The text of the url. (ie: "Learn more")
    uri: The link itself.
  """
    text = _messages.StringField(1)
    uri = _messages.StringField(2)