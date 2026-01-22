from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExactMatchResults(_messages.Message):
    """Results for exact match metric.

  Fields:
    exactMatchMetricValues: Output only. Exact match metric values.
  """
    exactMatchMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1ExactMatchMetricValue', 1, repeated=True)