from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DirectRawPredictResponse(_messages.Message):
    """Response message for PredictionService.DirectRawPredict.

  Fields:
    output: The prediction output.
  """
    output = _messages.BytesField(1)