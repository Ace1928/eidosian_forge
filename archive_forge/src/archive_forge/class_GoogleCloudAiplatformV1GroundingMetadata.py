from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GroundingMetadata(_messages.Message):
    """Metadata returned to client when grounding is enabled.

  Fields:
    groundingAttributions: Optional. List of grounding attributions.
    webSearchQueries: Optional. Web search queries for the following-up web
      search.
  """
    groundingAttributions = _messages.MessageField('GoogleCloudAiplatformV1GroundingAttribution', 1, repeated=True)
    webSearchQueries = _messages.StringField(2, repeated=True)