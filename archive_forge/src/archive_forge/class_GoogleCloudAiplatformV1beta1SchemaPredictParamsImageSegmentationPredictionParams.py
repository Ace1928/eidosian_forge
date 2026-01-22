from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaPredictParamsImageSegmentationPredictionParams(_messages.Message):
    """Prediction model parameters for Image Segmentation.

  Fields:
    confidenceThreshold: When the model predicts category of pixels of the
      image, it will only provide predictions for pixels that it is at least
      this much confident about. All other pixels will be classified as
      background. Default value is 0.5.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)