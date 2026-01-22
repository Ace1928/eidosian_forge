from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatestBackup(_messages.Message):
    """The details of the latest scheduled backup.

  Enums:
    StateValueValuesEnum: Output only. The current state of the backup.

  Fields:
    backupId: Output only. The ID of an in-progress scheduled backup. Empty if
      no backup is in progress.
    duration: Output only. The duration of the backup completion.
    startTime: Output only. The time when the backup was started.
    state: Output only. The current state of the backup.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the backup.

    Values:
      STATE_UNSPECIFIED: The state of the backup is unknown.
      IN_PROGRESS: The backup is in progress.
      SUCCEEDED: The backup completed.
      FAILED: The backup failed.
    """
        STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        SUCCEEDED = 2
        FAILED = 3
    backupId = _messages.StringField(1)
    duration = _messages.StringField(2)
    startTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)