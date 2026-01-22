from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictParamsImageClassificationPredictionParams(_messages.Message):
    """Prediction model parameters for Image Classification.

  Fields:
    confidenceThreshold: The Model only returns predictions with at least this
      confidence score. Default value is 0.0
    maxPredictions: The Model only returns up to that many top, by confidence
      score, predictions per instance. If this number is very high, the Model
      may return fewer predictions. Default value is 10.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    maxPredictions = _messages.IntegerField(2, variant=_messages.Variant.INT32)