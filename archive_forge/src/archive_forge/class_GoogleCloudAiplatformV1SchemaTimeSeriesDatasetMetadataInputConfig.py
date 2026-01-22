from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTimeSeriesDatasetMetadataInputConfig(_messages.Message):
    """The time series Dataset's data source. The Dataset doesn't store the
  data directly, but only pointer(s) to its data.

  Fields:
    bigquerySource: A
      GoogleCloudAiplatformV1SchemaTimeSeriesDatasetMetadataBigQuerySource
      attribute.
    gcsSource: A
      GoogleCloudAiplatformV1SchemaTimeSeriesDatasetMetadataGcsSource
      attribute.
  """
    bigquerySource = _messages.MessageField('GoogleCloudAiplatformV1SchemaTimeSeriesDatasetMetadataBigQuerySource', 1)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1SchemaTimeSeriesDatasetMetadataGcsSource', 2)