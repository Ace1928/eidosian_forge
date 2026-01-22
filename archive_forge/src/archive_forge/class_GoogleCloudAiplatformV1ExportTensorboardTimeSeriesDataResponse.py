from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportTensorboardTimeSeriesDataResponse(_messages.Message):
    """Response message for TensorboardService.ExportTensorboardTimeSeriesData.

  Fields:
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    timeSeriesDataPoints: The returned time series data points.
  """
    nextPageToken = _messages.StringField(1)
    timeSeriesDataPoints = _messages.MessageField('GoogleCloudAiplatformV1TimeSeriesDataPoint', 2, repeated=True)