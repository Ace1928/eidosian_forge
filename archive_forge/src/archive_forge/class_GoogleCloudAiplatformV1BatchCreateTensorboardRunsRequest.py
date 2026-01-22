from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchCreateTensorboardRunsRequest(_messages.Message):
    """Request message for TensorboardService.BatchCreateTensorboardRuns.

  Fields:
    requests: Required. The request message specifying the TensorboardRuns to
      create. A maximum of 1000 TensorboardRuns can be created in a batch.
  """
    requests = _messages.MessageField('GoogleCloudAiplatformV1CreateTensorboardRunRequest', 1, repeated=True)