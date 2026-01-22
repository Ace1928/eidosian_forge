from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadTensorboardTimeSeriesDataResponse(_messages.Message):
    """Response message for TensorboardService.ReadTensorboardTimeSeriesData.

  Fields:
    timeSeriesData: The returned time series data.
  """
    timeSeriesData = _messages.MessageField('GoogleCloudAiplatformV1beta1TimeSeriesData', 1)