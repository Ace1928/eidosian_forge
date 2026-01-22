from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NameAndKind(_messages.Message):
    """Basic metadata about a counter.

  Enums:
    KindValueValuesEnum: Counter aggregation kind.

  Fields:
    kind: Counter aggregation kind.
    name: Name of the counter.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Counter aggregation kind.

    Values:
      INVALID: Counter aggregation kind was not set.
      SUM: Aggregated value is the sum of all contributed values.
      MAX: Aggregated value is the max of all contributed values.
      MIN: Aggregated value is the min of all contributed values.
      MEAN: Aggregated value is the mean of all contributed values.
      OR: Aggregated value represents the logical 'or' of all contributed
        values.
      AND: Aggregated value represents the logical 'and' of all contributed
        values.
      SET: Aggregated value is a set of unique contributed values.
      DISTRIBUTION: Aggregated value captures statistics about a distribution.
      LATEST_VALUE: Aggregated value tracks the latest value of a variable.
    """
        INVALID = 0
        SUM = 1
        MAX = 2
        MIN = 3
        MEAN = 4
        OR = 5
        AND = 6
        SET = 7
        DISTRIBUTION = 8
        LATEST_VALUE = 9
    kind = _messages.EnumField('KindValueValuesEnum', 1)
    name = _messages.StringField(2)