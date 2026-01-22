from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootThresholdType(_messages.Message):
    """The type of score that bundled with a threshold, and will not be
  attending the final score calculation. How each score type uses the
  threshold can be implementation details.

  Enums:
    ScoreTypeValueValuesEnum:

  Fields:
    scoreType: A ScoreTypeValueValuesEnum attribute.
    threshold: A number attribute.
  """

    class ScoreTypeValueValuesEnum(_messages.Enum):
        """ScoreTypeValueValuesEnum enum type.

    Values:
      TYPE_UNKNOWN: Unknown scorer type.
      TYPE_SAFE: Safety scorer.
      TYPE_POLICY: Policy scorer.
      TYPE_GENERATION: Generation scorer.
    """
        TYPE_UNKNOWN = 0
        TYPE_SAFE = 1
        TYPE_POLICY = 2
        TYPE_GENERATION = 3
    scoreType = _messages.EnumField('ScoreTypeValueValuesEnum', 1)
    threshold = _messages.FloatField(2)