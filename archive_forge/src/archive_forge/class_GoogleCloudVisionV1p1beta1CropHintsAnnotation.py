from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1CropHintsAnnotation(_messages.Message):
    """Set of crop hints that are used to generate new crops when serving
  images.

  Fields:
    cropHints: Crop hint results.
  """
    cropHints = _messages.MessageField('GoogleCloudVisionV1p1beta1CropHint', 1, repeated=True)