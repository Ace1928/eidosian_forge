from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansBackupsGetBackupIndexDownloadUrlRequest(_messages.Message):
    """A
  GkebackupProjectsLocationsBackupPlansBackupsGetBackupIndexDownloadUrlRequest
  object.

  Fields:
    backup: Required. Full name of Backup resource. Format: projects/{project}
      /locations/{location}/backupPlans/{backup_plan}/backups/{backup}
  """
    backup = _messages.StringField(1, required=True)