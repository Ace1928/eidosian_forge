from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictInstanceImageObjectDetectionPredictionInstance(_messages.Message):
    """Prediction input format for Image Object Detection.

  Fields:
    content: The image bytes or Cloud Storage URI to make the prediction on.
    mimeType: The MIME type of the content of the image. Only the images in
      below listed MIME types are supported. - image/jpeg - image/gif -
      image/png - image/webp - image/bmp - image/tiff -
      image/vnd.microsoft.icon
  """
    content = _messages.StringField(1)
    mimeType = _messages.StringField(2)