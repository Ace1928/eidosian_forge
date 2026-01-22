from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ValidationError(_messages.Message):
    """Represents a single validation error.

  Enums:
    SeverityValueValuesEnum: The severity of the error.

  Fields:
    entries: The names of the entries that the error is associated with.
      Format: - `projects//agent`, if the error is associated with the entire
      agent. - `projects//agent/intents/`, if the error is associated with
      certain intents. - `projects//agent/intents//trainingPhrases/`, if the
      error is associated with certain intent training phrases. -
      `projects//agent/intents//parameters/`, if the error is associated with
      certain intent parameters. - `projects//agent/entities/`, if the error
      is associated with certain entities.
    errorMessage: The detailed error message.
    severity: The severity of the error.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the error.

    Values:
      SEVERITY_UNSPECIFIED: Not specified. This value should never be used.
      INFO: The agent doesn't follow Dialogflow best practices.
      WARNING: The agent may not behave as expected.
      ERROR: The agent may experience partial failures.
      CRITICAL: The agent may completely fail.
    """
        SEVERITY_UNSPECIFIED = 0
        INFO = 1
        WARNING = 2
        ERROR = 3
        CRITICAL = 4
    entries = _messages.StringField(1, repeated=True)
    errorMessage = _messages.StringField(2)
    severity = _messages.EnumField('SeverityValueValuesEnum', 3)