from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationCheckResult(_messages.Message):
    """ValidationCheckResult defines the details about the validation check.

  Enums:
    StateValueValuesEnum: The validation check state.

  Fields:
    category: The category of the validation.
    description: The description of the validation check.
    details: Detailed failure information, which might be unformatted.
    reason: A human-readable message of the check failure.
    state: The validation check state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The validation check state.

    Values:
      STATE_UNKNOWN: The default value. The check result is unknown.
      STATE_FAILURE: The check failed.
      STATE_SKIPPED: The check was skipped.
      STATE_FATAL: The check itself failed to complete.
      STATE_WARNING: The check encountered a warning.
    """
        STATE_UNKNOWN = 0
        STATE_FAILURE = 1
        STATE_SKIPPED = 2
        STATE_FATAL = 3
        STATE_WARNING = 4
    category = _messages.StringField(1)
    description = _messages.StringField(2)
    details = _messages.StringField(3)
    reason = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)