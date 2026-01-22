from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesBackupSchedulesPatchRequest(_messages.Message):
    """A FirestoreProjectsDatabasesBackupSchedulesPatchRequest object.

  Fields:
    googleFirestoreAdminV1BackupSchedule: A
      GoogleFirestoreAdminV1BackupSchedule resource to be passed as the
      request body.
    name: Output only. The unique backup schedule identifier across all
      locations and databases for the given project. This will be auto-
      assigned. Format is `projects/{project}/databases/{database}/backupSched
      ules/{backup_schedule}`
    updateMask: The list of fields to be updated.
  """
    googleFirestoreAdminV1BackupSchedule = _messages.MessageField('GoogleFirestoreAdminV1BackupSchedule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)