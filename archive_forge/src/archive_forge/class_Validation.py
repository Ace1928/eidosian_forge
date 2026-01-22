from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Validation(_messages.Message):
    """A validation to perform on a stream.

  Enums:
    StateValueValuesEnum: Output only. Validation execution status.

  Fields:
    code: A custom code identifying this validation.
    description: A short description of the validation.
    message: Messages reflecting the validation results.
    state: Output only. Validation execution status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Validation execution status.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      NOT_EXECUTED: Validation did not execute.
      FAILED: Validation failed.
      PASSED: Validation passed.
      WARNING: Validation executed with warnings.
    """
        STATE_UNSPECIFIED = 0
        NOT_EXECUTED = 1
        FAILED = 2
        PASSED = 3
        WARNING = 4
    code = _messages.StringField(1)
    description = _messages.StringField(2)
    message = _messages.MessageField('ValidationMessage', 3, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 4)