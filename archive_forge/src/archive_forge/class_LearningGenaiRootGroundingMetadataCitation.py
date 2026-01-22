from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootGroundingMetadataCitation(_messages.Message):
    """A LearningGenaiRootGroundingMetadataCitation object.

  Fields:
    endIndex: Index in the prediction output where the citation ends
      (exclusive). Must be > start_index and <= len(output).
    factIndex: Index of the fact supporting this claim. Should be within the
      range of the `world_facts` in the GenerateResponse.
    score: Confidence score of this entailment. Value is [0,1] with 1 is the
      most confidence.
    startIndex: Index in the prediction output where the citation starts
      (inclusive). Must be >= 0 and < end_index.
  """
    endIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    factIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    score = _messages.FloatField(3)
    startIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)