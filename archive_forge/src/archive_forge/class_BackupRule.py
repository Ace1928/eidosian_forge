from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupRule(_messages.Message):
    """`BackupRule` binds the backup schedule to a retention policy.

  Fields:
    backupRetentionDays: Required. Configures the duration for which backup
      data will be kept. It is defined in "days". The value should be greater
      than or equal to minimum enforced retention of the backup vault.
    backupVault: Required. Resource name of backup vault which will be used as
      storage location for backups. Format:
      projects/{project}/locations/{location}/backupVaults/{backupvault}
    backupVaultServiceAccount: Output only. The Google Cloud Platform Service
      Account to be used by the BackupVault for taking backups. Specify the
      email address of the Backup Vault Service Account. (precedent: https://s
      ource.corp.google.com/piper///depot/google3/google/container/v1/cluster_
      service.proto;l=1014-1019)
    displayName: Optional. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. The display name of the
      `BackupRule`.
    ruleId: Required. Immutable. The unique id of this `BackupRule`. The
      `rule_id` is unique per `BackupPlan`.The `rule_id` must start with a
      lowercase letter followed by up to 62 lowercase letters, numbers, or
      hyphens. Pattern, /a-z{,62}/.
    standardSchedule: Required. Defines a schedule that runs within the
      confines of a defined window of time.
  """
    backupRetentionDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    backupVault = _messages.StringField(2)
    backupVaultServiceAccount = _messages.StringField(3)
    displayName = _messages.StringField(4)
    ruleId = _messages.StringField(5)
    standardSchedule = _messages.MessageField('StandardSchedule', 6)