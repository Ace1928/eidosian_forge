from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SamplingStrategyRandomSampleConfig(_messages.Message):
    """Requests are randomly selected.

  Fields:
    sampleRate: Sample rate (0, 1]
  """
    sampleRate = _messages.FloatField(1)