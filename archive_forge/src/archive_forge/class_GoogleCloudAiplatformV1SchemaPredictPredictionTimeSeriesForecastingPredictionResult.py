from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionTimeSeriesForecastingPredictionResult(_messages.Message):
    """Prediction output format for Time Series Forecasting.

  Fields:
    quantilePredictions: Quantile predictions, in 1-1 correspondence with
      quantile_values.
    quantileValues: Quantile values.
    tftFeatureImportance: Only use these if TFt is enabled.
    value: The regression value.
  """
    quantilePredictions = _messages.FloatField(1, repeated=True, variant=_messages.Variant.FLOAT)
    quantileValues = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)
    tftFeatureImportance = _messages.MessageField('GoogleCloudAiplatformV1SchemaPredictPredictionTftFeatureImportance', 3)
    value = _messages.FloatField(4, variant=_messages.Variant.FLOAT)