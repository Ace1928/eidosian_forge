from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchCreateTensorboardTimeSeriesResponse(_messages.Message):
    """Response message for
  TensorboardService.BatchCreateTensorboardTimeSeries.

  Fields:
    tensorboardTimeSeries: The created TensorboardTimeSeries.
  """
    tensorboardTimeSeries = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardTimeSeries', 1, repeated=True)