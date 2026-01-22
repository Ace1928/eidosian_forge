from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolNameMatchResults(_messages.Message):
    """Results for tool name match metric.

  Fields:
    toolNameMatchMetricValues: Output only. Tool name match metric values.
  """
    toolNameMatchMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolNameMatchMetricValue', 1, repeated=True)