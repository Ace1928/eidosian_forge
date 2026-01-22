from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RefersToValueValuesEnum(_messages.Enum):
    """Reference to which the message applies.

    Values:
      UNSPECIFIED: Status doesn't refer to any particular input.
      BREAKPOINT_SOURCE_LOCATION: Status applies to the breakpoint and is
        related to its location.
      BREAKPOINT_CONDITION: Status applies to the breakpoint and is related to
        its condition.
      BREAKPOINT_EXPRESSION: Status applies to the breakpoint and is related
        to its expressions.
      BREAKPOINT_AGE: Status applies to the breakpoint and is related to its
        age.
      BREAKPOINT_CANARY_FAILED: Status applies to the breakpoint when the
        breakpoint failed to exit the canary state.
      VARIABLE_NAME: Status applies to the entire variable.
      VARIABLE_VALUE: Status applies to variable value (variable name is
        valid).
    """
    UNSPECIFIED = 0
    BREAKPOINT_SOURCE_LOCATION = 1
    BREAKPOINT_CONDITION = 2
    BREAKPOINT_EXPRESSION = 3
    BREAKPOINT_AGE = 4
    BREAKPOINT_CANARY_FAILED = 5
    VARIABLE_NAME = 6
    VARIABLE_VALUE = 7