from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessLevelsEvalStatesValueListEntryValuesEnum(_messages.Enum):
    """AccessLevelsEvalStatesValueListEntryValuesEnum enum type.

    Values:
      ACCESS_LEVEL_EVAL_STATE_UNSPECIFIED: Not used
      ACCESS_LEVEL_EVAL_STATE_SATISFIED: The access level is satisfied
      ACCESS_LEVEL_EVAL_STATE_UNSATISFIED: The access level is unsatisfied
      ACCESS_LEVEL_EVAL_STATE_ERROR: The access level is not satisfied nor
        unsatisfied
      ACCESS_LEVEL_EVAL_STATE_NOT_EXIST: The access level does not exist
    """
    ACCESS_LEVEL_EVAL_STATE_UNSPECIFIED = 0
    ACCESS_LEVEL_EVAL_STATE_SATISFIED = 1
    ACCESS_LEVEL_EVAL_STATE_UNSATISFIED = 2
    ACCESS_LEVEL_EVAL_STATE_ERROR = 3
    ACCESS_LEVEL_EVAL_STATE_NOT_EXIST = 4