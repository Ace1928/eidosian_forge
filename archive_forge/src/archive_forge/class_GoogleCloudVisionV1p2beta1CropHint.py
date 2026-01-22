from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1CropHint(_messages.Message):
    """Single crop hint that is used to generate a new crop when serving an
  image.

  Fields:
    boundingPoly: The bounding polygon for the crop region. The coordinates of
      the bounding box are in the original image's scale.
    confidence: Confidence of this being a salient region. Range [0, 1].
    importanceFraction: Fraction of importance of this salient region with
      respect to the original image.
  """
    boundingPoly = _messages.MessageField('GoogleCloudVisionV1p2beta1BoundingPoly', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    importanceFraction = _messages.FloatField(3, variant=_messages.Variant.FLOAT)