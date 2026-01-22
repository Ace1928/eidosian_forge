from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1ColorInfo(_messages.Message):
    """Color information consists of RGB channels, score, and the fraction of
  the image that the color occupies in the image.

  Fields:
    color: RGB components of the color.
    pixelFraction: The fraction of pixels the color occupies in the image.
      Value in range [0, 1].
    score: Image-specific score for this color. Value in range [0, 1].
  """
    color = _messages.MessageField('Color', 1)
    pixelFraction = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)