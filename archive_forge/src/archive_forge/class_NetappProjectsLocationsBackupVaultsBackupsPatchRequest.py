from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsBackupsPatchRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsBackupsPatchRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: Identifier. The resource name of the backup. Format: `projects/{proj
      ect_id}/locations/{location}/backupVaults/{backup_vault_id}/backups/{bac
      kup_id}`.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Backup resource to be updated. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)