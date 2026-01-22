from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolCallValidResults(_messages.Message):
    """Results for tool call valid metric.

  Fields:
    toolCallValidMetricValues: Output only. Tool call valid metric values.
  """
    toolCallValidMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolCallValidMetricValue', 1, repeated=True)