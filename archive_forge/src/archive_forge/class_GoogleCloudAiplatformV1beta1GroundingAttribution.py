from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GroundingAttribution(_messages.Message):
    """Grounding attribution.

  Fields:
    confidenceScore: Optional. Output only. Confidence score of the
      attribution. Ranges from 0 to 1. 1 is the most confident.
    retrievedContext: Optional. Attribution from context retrieved by the
      retrieval tools.
    segment: Output only. Segment of the content this attribution belongs to.
    web: Optional. Attribution from the web.
  """
    confidenceScore = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    retrievedContext = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundingAttributionRetrievedContext', 2)
    segment = _messages.MessageField('GoogleCloudAiplatformV1beta1Segment', 3)
    web = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundingAttributionWeb', 4)