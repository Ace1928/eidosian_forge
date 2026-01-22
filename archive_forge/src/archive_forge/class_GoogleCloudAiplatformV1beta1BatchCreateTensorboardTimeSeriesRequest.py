from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchCreateTensorboardTimeSeriesRequest(_messages.Message):
    """Request message for TensorboardService.BatchCreateTensorboardTimeSeries.

  Fields:
    requests: Required. The request message specifying the
      TensorboardTimeSeries to create. A maximum of 1000 TensorboardTimeSeries
      can be created in a batch.
  """
    requests = _messages.MessageField('GoogleCloudAiplatformV1beta1CreateTensorboardTimeSeriesRequest', 1, repeated=True)