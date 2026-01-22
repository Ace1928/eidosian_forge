from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1DominantColorsAnnotation(_messages.Message):
    """Set of dominant colors and their corresponding scores.

  Fields:
    colors: RGB color values with their score and pixel fraction.
  """
    colors = _messages.MessageField('GoogleCloudVisionV1p2beta1ColorInfo', 1, repeated=True)