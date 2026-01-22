from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Neighbor(_messages.Message):
    """Neighbors for example-based explanations.

  Fields:
    neighborDistance: Output only. The neighbor distance.
    neighborId: Output only. The neighbor id.
  """
    neighborDistance = _messages.FloatField(1)
    neighborId = _messages.StringField(2)