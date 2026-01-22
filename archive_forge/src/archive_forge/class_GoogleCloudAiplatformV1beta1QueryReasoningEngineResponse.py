from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QueryReasoningEngineResponse(_messages.Message):
    """Response message for ReasoningEngineExecutionService.Query

  Fields:
    output: Response provided by users in JSON object format.
  """
    output = _messages.MessageField('extra_types.JsonValue', 1)