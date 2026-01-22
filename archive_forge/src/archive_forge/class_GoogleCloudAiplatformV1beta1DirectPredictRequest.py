from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DirectPredictRequest(_messages.Message):
    """Request message for PredictionService.DirectPredict.

  Fields:
    inputs: The prediction input.
    parameters: The parameters that govern the prediction.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1Tensor', 1, repeated=True)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1Tensor', 2)