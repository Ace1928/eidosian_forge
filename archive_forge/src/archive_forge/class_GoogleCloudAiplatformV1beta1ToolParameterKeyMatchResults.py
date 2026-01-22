from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolParameterKeyMatchResults(_messages.Message):
    """Results for tool parameter key match metric.

  Fields:
    toolParameterKeyMatchMetricValues: Output only. Tool parameter key match
      metric values.
  """
    toolParameterKeyMatchMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKeyMatchMetricValue', 1, repeated=True)