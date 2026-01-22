from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ListBackupSchedulesResponse(_messages.Message):
    """The response for FirestoreAdmin.ListBackupSchedules.

  Fields:
    backupSchedules: List of all backup schedules.
  """
    backupSchedules = _messages.MessageField('GoogleFirestoreAdminV1BackupSchedule', 1, repeated=True)