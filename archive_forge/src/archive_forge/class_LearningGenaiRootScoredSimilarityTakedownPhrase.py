from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootScoredSimilarityTakedownPhrase(_messages.Message):
    """Proto containing the results from the Universal Sentence Encoder / Other
  models

  Fields:
    phrase: A LearningGenaiRootSimilarityTakedownPhrase attribute.
    similarityScore: A number attribute.
  """
    phrase = _messages.MessageField('LearningGenaiRootSimilarityTakedownPhrase', 1)
    similarityScore = _messages.FloatField(2, variant=_messages.Variant.FLOAT)