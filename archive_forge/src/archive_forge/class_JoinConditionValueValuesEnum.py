from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JoinConditionValueValuesEnum(_messages.Enum):
    """Optional. Specifies if a join is required or not on queries for the
    view. Default is JOIN_CONDITION_UNSPECIFIED.

    Values:
      JOIN_CONDITION_UNSPECIFIED: A join is neither required nor restricted on
        any column. Default value.
      JOIN_ANY: A join is required on at least one of the specified columns.
      JOIN_ALL: A join is required on all specified columns.
      JOIN_NOT_REQUIRED: A join is not required, but if present it is only
        permitted on 'join_allowed_columns'
      JOIN_BLOCKED: Joins are blocked for all queries.
    """
    JOIN_CONDITION_UNSPECIFIED = 0
    JOIN_ANY = 1
    JOIN_ALL = 2
    JOIN_NOT_REQUIRED = 3
    JOIN_BLOCKED = 4