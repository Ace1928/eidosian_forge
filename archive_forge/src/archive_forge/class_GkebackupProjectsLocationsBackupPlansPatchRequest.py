from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansPatchRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansPatchRequest object.

  Fields:
    backupPlan: A BackupPlan resource to be passed as the request body.
    name: Output only. The full name of the BackupPlan resource. Format:
      `projects/*/locations/*/backupPlans/*`
    updateMask: Optional. This is used to specify the fields to be overwritten
      in the BackupPlan targeted for update. The values for each of these
      updated fields will be taken from the `backup_plan` provided with this
      request. Field names are relative to the root of the resource (e.g.,
      `description`, `backup_config.include_volume_data`, etc.) If no
      `update_mask` is provided, all fields in `backup_plan` will be written
      to the target BackupPlan resource. Note that OUTPUT_ONLY and IMMUTABLE
      fields in `backup_plan` are ignored and are not used to update the
      target BackupPlan.
  """
    backupPlan = _messages.MessageField('BackupPlan', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)