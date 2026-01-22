from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootInternalMetadata(_messages.Message):
    """A LearningGenaiRootInternalMetadata object.

  Fields:
    scoredTokens: A LearningGenaiRootScoredToken attribute.
  """
    scoredTokens = _messages.MessageField('LearningGenaiRootScoredToken', 1, repeated=True)