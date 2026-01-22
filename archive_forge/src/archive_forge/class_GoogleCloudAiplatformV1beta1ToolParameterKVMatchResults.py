from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolParameterKVMatchResults(_messages.Message):
    """Results for tool parameter key value match metric.

  Fields:
    toolParameterKvMatchMetricValues: Output only. Tool parameter key value
      match metric values.
  """
    toolParameterKvMatchMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolParameterKVMatchMetricValue', 1, repeated=True)