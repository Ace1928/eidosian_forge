from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ScoreComponentRecommendationAction(_messages.Message):
    """Action to improve security score.

  Fields:
    actionContext: Action context for the action.
    description: Description of the action.
  """
    actionContext = _messages.MessageField('GoogleCloudApigeeV1ScoreComponentRecommendationActionActionContext', 1)
    description = _messages.StringField(2)