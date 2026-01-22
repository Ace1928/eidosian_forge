from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRoutingDecisionMetadataTokenLengthBasedModelMaxTokenMetadata(_messages.Message):
    """A LearningGenaiRootRoutingDecisionMetadataTokenLengthBasedModelMaxTokenM
  etadata object.

  Fields:
    maxNumInputTokens: A integer attribute.
    maxNumOutputTokens: A integer attribute.
    modelId: A string attribute.
  """
    maxNumInputTokens = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxNumOutputTokens = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    modelId = _messages.StringField(3)