from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootTakedownResult(_messages.Message):
    """A LearningGenaiRootTakedownResult object.

  Fields:
    allowed: False when query or response should be taken down by any of the
      takedown rules, true otherwise.
    regexTakedownResult: A LearningGenaiRootRegexTakedownResult attribute.
    requestResponseTakedownResult: A
      LearningGenaiRootRequestResponseTakedownResult attribute.
    similarityTakedownResult: A LearningGenaiRootSimilarityTakedownResult
      attribute.
  """
    allowed = _messages.BooleanField(1)
    regexTakedownResult = _messages.MessageField('LearningGenaiRootRegexTakedownResult', 2)
    requestResponseTakedownResult = _messages.MessageField('LearningGenaiRootRequestResponseTakedownResult', 3)
    similarityTakedownResult = _messages.MessageField('LearningGenaiRootSimilarityTakedownResult', 4)