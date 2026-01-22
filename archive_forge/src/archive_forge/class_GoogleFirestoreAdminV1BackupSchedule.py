from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1BackupSchedule(_messages.Message):
    """A backup schedule for a Cloud Firestore Database. This resource is owned
  by the database it is backing up, and is deleted along with the database.
  The actual backups are not though.

  Fields:
    createTime: Output only. The timestamp at which this backup schedule was
      created and effective since. No backups will be created for this
      schedule before this time.
    dailyRecurrence: For a schedule that runs daily.
    name: Output only. The unique backup schedule identifier across all
      locations and databases for the given project. This will be auto-
      assigned. Format is `projects/{project}/databases/{database}/backupSched
      ules/{backup_schedule}`
    retention: At what relative time in the future, compared to its creation
      time, the backup should be deleted, e.g. keep backups for 7 days.
    updateTime: Output only. The timestamp at which this backup schedule was
      most recently updated. When a backup schedule is first created, this is
      the same as create_time.
    weeklyRecurrence: For a schedule that runs weekly on a specific day.
  """
    createTime = _messages.StringField(1)
    dailyRecurrence = _messages.MessageField('GoogleFirestoreAdminV1DailyRecurrence', 2)
    name = _messages.StringField(3)
    retention = _messages.StringField(4)
    updateTime = _messages.StringField(5)
    weeklyRecurrence = _messages.MessageField('GoogleFirestoreAdminV1WeeklyRecurrence', 6)