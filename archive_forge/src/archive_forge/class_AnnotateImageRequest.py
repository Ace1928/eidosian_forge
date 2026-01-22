from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotateImageRequest(_messages.Message):
    """Request for performing Google Cloud Vision API tasks over a user-
  provided image, with user-requested features, and with context information.

  Fields:
    features: Requested features.
    image: The image to be processed.
    imageContext: Additional context that may accompany the image.
  """
    features = _messages.MessageField('Feature', 1, repeated=True)
    image = _messages.MessageField('Image', 2)
    imageContext = _messages.MessageField('ImageContext', 3)