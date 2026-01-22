from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JoinRestrictionPolicy(_messages.Message):
    """Represents privacy policy associated with "join restrictions". Join
  restriction gives data providers the ability to enforce joins on the
  'join_allowed_columns' when data is queried from a privacy protected view.

  Enums:
    JoinConditionValueValuesEnum: Optional. Specifies if a join is required or
      not on queries for the view. Default is JOIN_CONDITION_UNSPECIFIED.

  Fields:
    joinAllowedColumns: Optional. The only columns that joins are allowed on.
      This field is must be specified for join_conditions JOIN_ANY and
      JOIN_ALL and it cannot be set for JOIN_BLOCKED.
    joinCondition: Optional. Specifies if a join is required or not on queries
      for the view. Default is JOIN_CONDITION_UNSPECIFIED.
  """

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
    joinAllowedColumns = _messages.StringField(1, repeated=True)
    joinCondition = _messages.EnumField('JoinConditionValueValuesEnum', 2)