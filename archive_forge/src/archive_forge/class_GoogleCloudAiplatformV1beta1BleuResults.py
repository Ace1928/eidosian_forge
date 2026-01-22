from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BleuResults(_messages.Message):
    """Results for bleu metric.

  Fields:
    bleuMetricValues: Output only. Bleu metric values.
  """
    bleuMetricValues = _messages.MessageField('GoogleCloudAiplatformV1beta1BleuMetricValue', 1, repeated=True)