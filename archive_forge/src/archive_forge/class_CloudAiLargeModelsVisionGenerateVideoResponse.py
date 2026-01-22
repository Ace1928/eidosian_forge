from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionGenerateVideoResponse(_messages.Message):
    """Generate video response.

  Fields:
    generatedSamples: The generates samples.
    raiMediaFilteredCount: Returns if any videos were filtered due to RAI
      policies.
    raiMediaFilteredReasons: Returns rai failure reasons if any.
    raiTextFilteredReason: Returns filtered text rai info.
  """
    generatedSamples = _messages.MessageField('CloudAiLargeModelsVisionMedia', 1, repeated=True)
    raiMediaFilteredCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    raiMediaFilteredReasons = _messages.StringField(3, repeated=True)
    raiTextFilteredReason = _messages.MessageField('CloudAiLargeModelsVisionFilteredText', 4)