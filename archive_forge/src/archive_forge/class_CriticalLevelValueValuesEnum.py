from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CriticalLevelValueValuesEnum(_messages.Enum):
    """If it is a critical failed node that blocks the expected state of this
    access level. It is valid only for boolean expression nodes and when the
    node's expected value doesn't equal to actual value

    Values:
      CRITICAL_LEVEL_UNSPECIFIED: reserved
      CRITICAL_LEVEL_LOW: The node is not on the critical path to the expected
        state of this access level.
      CRITICAL_LEVEL_HIGH: The node is on the critical path to the expected
        state of this access level.
    """
    CRITICAL_LEVEL_UNSPECIFIED = 0
    CRITICAL_LEVEL_LOW = 1
    CRITICAL_LEVEL_HIGH = 2