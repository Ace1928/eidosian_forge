from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p2beta1DetectedLandmark(_messages.Message):
    """A generic detected landmark represented by name in string format and a
  2D location.

  Fields:
    confidence: The confidence score of the detected landmark. Range [0, 1].
    name: The name of this landmark, for example, left_hand, right_shoulder.
    point: The 2D point of the detected landmark using the normalized image
      coordindate system. The normalized coordinates have the range from 0 to
      1.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    name = _messages.StringField(2)
    point = _messages.MessageField('GoogleCloudVideointelligenceV1p2beta1NormalizedVertex', 3)