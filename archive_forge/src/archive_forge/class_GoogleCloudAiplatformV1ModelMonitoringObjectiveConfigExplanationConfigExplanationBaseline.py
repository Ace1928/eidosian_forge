from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelMonitoringObjectiveConfigExplanationConfigExplanationBaseline(_messages.Message):
    """Output from BatchPredictionJob for Model Monitoring baseline dataset,
  which can be used to generate baseline attribution scores.

  Enums:
    PredictionFormatValueValuesEnum: The storage format of the predictions
      generated BatchPrediction job.

  Fields:
    bigquery: BigQuery location for BatchExplain output.
    gcs: Cloud Storage location for BatchExplain output.
    predictionFormat: The storage format of the predictions generated
      BatchPrediction job.
  """

    class PredictionFormatValueValuesEnum(_messages.Enum):
        """The storage format of the predictions generated BatchPrediction job.

    Values:
      PREDICTION_FORMAT_UNSPECIFIED: Should not be set.
      JSONL: Predictions are in JSONL files.
      BIGQUERY: Predictions are in BigQuery.
    """
        PREDICTION_FORMAT_UNSPECIFIED = 0
        JSONL = 1
        BIGQUERY = 2
    bigquery = _messages.MessageField('GoogleCloudAiplatformV1BigQueryDestination', 1)
    gcs = _messages.MessageField('GoogleCloudAiplatformV1GcsDestination', 2)
    predictionFormat = _messages.EnumField('PredictionFormatValueValuesEnum', 3)