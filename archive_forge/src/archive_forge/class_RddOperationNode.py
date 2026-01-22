from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddOperationNode(_messages.Message):
    """A node in the RDD operation graph. Corresponds to a single RDD.

  Enums:
    OutputDeterministicLevelValueValuesEnum:

  Fields:
    barrier: A boolean attribute.
    cached: A boolean attribute.
    callsite: A string attribute.
    name: A string attribute.
    nodeId: A integer attribute.
    outputDeterministicLevel: A OutputDeterministicLevelValueValuesEnum
      attribute.
  """

    class OutputDeterministicLevelValueValuesEnum(_messages.Enum):
        """OutputDeterministicLevelValueValuesEnum enum type.

    Values:
      DETERMINISTIC_LEVEL_UNSPECIFIED: <no description>
      DETERMINISTIC_LEVEL_DETERMINATE: <no description>
      DETERMINISTIC_LEVEL_UNORDERED: <no description>
      DETERMINISTIC_LEVEL_INDETERMINATE: <no description>
    """
        DETERMINISTIC_LEVEL_UNSPECIFIED = 0
        DETERMINISTIC_LEVEL_DETERMINATE = 1
        DETERMINISTIC_LEVEL_UNORDERED = 2
        DETERMINISTIC_LEVEL_INDETERMINATE = 3
    barrier = _messages.BooleanField(1)
    cached = _messages.BooleanField(2)
    callsite = _messages.StringField(3)
    name = _messages.StringField(4)
    nodeId = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    outputDeterministicLevel = _messages.EnumField('OutputDeterministicLevelValueValuesEnum', 6)