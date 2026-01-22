from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsCreateRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsCreateRequest object.

  Fields:
    backupVault: A BackupVault resource to be passed as the request body.
    backupVaultId: Required. The ID to use for the backupVault. The ID must be
      unique within the specified location. The max supported length is 63
      characters. This value must start with a lowercase letter followed by up
      to 62 lowercase letters, numbers, or hyphens, and cannot end with a
      hyphen. Values that do not match this pattern will trigger an
      INVALID_ARGUMENT error.
    parent: Required. The location to create the backup vaults, in the format
      `projects/{project_id}/locations/{location}`
  """
    backupVault = _messages.MessageField('BackupVault', 1)
    backupVaultId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)