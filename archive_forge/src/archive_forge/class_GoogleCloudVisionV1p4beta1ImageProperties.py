from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1ImageProperties(_messages.Message):
    """Stores image properties, such as dominant colors.

  Fields:
    dominantColors: If present, dominant colors completed successfully.
  """
    dominantColors = _messages.MessageField('GoogleCloudVisionV1p4beta1DominantColorsAnnotation', 1)