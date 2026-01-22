from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainBackupRun(_messages.Message):
    """A backup run.

  Enums:
    StatusValueValuesEnum: The status of this run. REQUIRED

  Fields:
    endTime: The time the backup operation completed. REQUIRED
    error: Information about why the backup operation failed. This is only
      present if the run has the FAILED status. OPTIONAL
    startTime: The time the backup operation started. REQUIRED
    status: The status of this run. REQUIRED
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The status of this run. REQUIRED

    Values:
      STATUS_UNSPECIFIED: <no description>
      SUCCESSFUL: The backup was successful.
      FAILED: The backup was unsuccessful.
    """
        STATUS_UNSPECIFIED = 0
        SUCCESSFUL = 1
        FAILED = 2
    endTime = _messages.StringField(1)
    error = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainOperationError', 2)
    startTime = _messages.StringField(3)
    status = _messages.EnumField('StatusValueValuesEnum', 4)