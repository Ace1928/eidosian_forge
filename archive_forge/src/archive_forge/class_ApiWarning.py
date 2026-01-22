from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApiWarning(_messages.Message):
    """An Admin API warning message.

  Enums:
    CodeValueValuesEnum: Code to uniquely identify the warning type.

  Fields:
    code: Code to uniquely identify the warning type.
    message: The warning message.
    region: The region name for REGION_UNREACHABLE warning.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Code to uniquely identify the warning type.

    Values:
      SQL_API_WARNING_CODE_UNSPECIFIED: An unknown or unset warning type from
        Cloud SQL API.
      REGION_UNREACHABLE: Warning when one or more regions are not reachable.
        The returned result set may be incomplete.
      MAX_RESULTS_EXCEEDS_LIMIT: Warning when user provided maxResults
        parameter exceeds the limit. The returned result set may be
        incomplete.
      COMPROMISED_CREDENTIALS: Warning when user tries to create/update a user
        with credentials that have previously been compromised by a public
        data breach.
      INTERNAL_STATE_FAILURE: Warning when the operation succeeds but some
        non-critical workflow state failed.
    """
        SQL_API_WARNING_CODE_UNSPECIFIED = 0
        REGION_UNREACHABLE = 1
        MAX_RESULTS_EXCEEDS_LIMIT = 2
        COMPROMISED_CREDENTIALS = 3
        INTERNAL_STATE_FAILURE = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    message = _messages.StringField(2)
    region = _messages.StringField(3)