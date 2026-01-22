from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansBackupsDeleteRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansBackupsDeleteRequest object.

  Fields:
    etag: Optional. If provided, this value must match the current value of
      the target Backup's etag field or the request is rejected.
    force: Optional. If set to true, any VolumeBackups below this Backup will
      also be deleted. Otherwise, the request will only succeed if the Backup
      has no VolumeBackups.
    name: Required. Name of the Backup resource. Format:
      `projects/*/locations/*/backupPlans/*/backups/*`
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)