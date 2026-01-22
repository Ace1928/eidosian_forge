from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1RestoreDatabaseRequest(_messages.Message):
    """The request message for FirestoreAdmin.RestoreDatabase.

  Fields:
    backup: Backup to restore from. Must be from the same project as the
      parent. Format is:
      `projects/{project_id}/locations/{location}/backups/{backup}`
    databaseId: Required. The ID to use for the database, which will become
      the final component of the database's resource name. This database id
      must not be associated with an existing database. This value should be
      4-63 characters. Valid characters are /a-z-/ with first character a
      letter and the last a letter or a number. Must not be UUID-like
      /[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}/. "(default)" database id is
      also valid.
    databaseSnapshot: Database snapshot to restore from. The source database
      must exist and have enabled PITR. The restored database will be created
      in the same location as the source database.
  """
    backup = _messages.StringField(1)
    databaseId = _messages.StringField(2)
    databaseSnapshot = _messages.MessageField('GoogleFirestoreAdminV1DatabaseSnapshot', 3)