from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ScoreComponent(_messages.Message):
    """Component is an individual security element that is scored.

  Fields:
    calculateTime: Time when score was calculated.
    dataCaptureTime: Time in the requested time period when data was last
      captured to compute the score.
    drilldownPaths: List of paths for next components.
    recommendations: List of recommendations to improve API security.
    score: Score for the component.
    scorePath: Path of the component. Example:
      /org@myorg/envgroup@myenvgroup/proxies/proxy@myproxy
  """
    calculateTime = _messages.StringField(1)
    dataCaptureTime = _messages.StringField(2)
    drilldownPaths = _messages.StringField(3, repeated=True)
    recommendations = _messages.MessageField('GoogleCloudApigeeV1ScoreComponentRecommendation', 4, repeated=True)
    score = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    scorePath = _messages.StringField(6)