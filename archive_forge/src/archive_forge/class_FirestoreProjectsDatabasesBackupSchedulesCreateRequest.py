from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesBackupSchedulesCreateRequest(_messages.Message):
    """A FirestoreProjectsDatabasesBackupSchedulesCreateRequest object.

  Fields:
    googleFirestoreAdminV1BackupSchedule: A
      GoogleFirestoreAdminV1BackupSchedule resource to be passed as the
      request body.
    parent: Required. The parent database. Format
      `projects/{project}/databases/{database}`
  """
    googleFirestoreAdminV1BackupSchedule = _messages.MessageField('GoogleFirestoreAdminV1BackupSchedule', 1)
    parent = _messages.StringField(2, required=True)