from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ScoreComponentRecommendationActionActionContext(_messages.Message):
    """Action context are all the relevant details for the action.

  Fields:
    documentationLink: Documentation link for the action.
  """
    documentationLink = _messages.StringField(1)