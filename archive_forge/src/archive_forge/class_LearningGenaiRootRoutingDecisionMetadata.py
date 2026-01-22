from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRoutingDecisionMetadata(_messages.Message):
    """Debug metadata about the routing decision.

  Fields:
    scoreBasedRoutingMetadata: A
      LearningGenaiRootRoutingDecisionMetadataScoreBased attribute.
    tokenLengthBasedRoutingMetadata: A
      LearningGenaiRootRoutingDecisionMetadataTokenLengthBased attribute.
  """
    scoreBasedRoutingMetadata = _messages.MessageField('LearningGenaiRootRoutingDecisionMetadataScoreBased', 1)
    tokenLengthBasedRoutingMetadata = _messages.MessageField('LearningGenaiRootRoutingDecisionMetadataTokenLengthBased', 2)