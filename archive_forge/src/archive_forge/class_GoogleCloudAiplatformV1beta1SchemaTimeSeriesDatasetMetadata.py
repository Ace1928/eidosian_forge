from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTimeSeriesDatasetMetadata(_messages.Message):
    """The metadata of Datasets that contain time series data.

  Fields:
    inputConfig: A
      GoogleCloudAiplatformV1beta1SchemaTimeSeriesDatasetMetadataInputConfig
      attribute.
    timeColumn: The column name of the time column that identifies time order
      in the time series.
    timeSeriesIdentifierColumn: The column name of the time series identifier
      column that identifies the time series.
  """
    inputConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTimeSeriesDatasetMetadataInputConfig', 1)
    timeColumn = _messages.StringField(2)
    timeSeriesIdentifierColumn = _messages.StringField(3)