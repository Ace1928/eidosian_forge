from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1RecognizedCelebrity(_messages.Message):
    """The recognized celebrity with confidence score.

  Fields:
    celebrity: The recognized celebrity.
    confidence: Recognition confidence. Range [0, 1].
  """
    celebrity = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1Celebrity', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)