from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceBackup(_messages.Message):
    """The details of a backup resource.

  Enums:
    StateValueValuesEnum: Output only. The current state of the backup.

  Fields:
    createTime: Output only. The time when the backup was started.
    encryptionConfig: Output only. Current status of the CMEK encryption
    expireTime: Output only. The time when the backup will be deleted.
    name: Immutable. The relative resource name of the backup, in the
      following form: `projects/{project_number}/locations/{location_id}/insta
      nces/{instance_id}/backups/{backup}`
    state: Output only. The current state of the backup.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the backup.

    Values:
      STATE_UNSPECIFIED: The state of the backup is unknown.
      CREATING: The backup is being created.
      DELETING: The backup is being deleted.
      ACTIVE: The backup is active and ready to use.
      FAILED: The backup failed.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        DELETING = 2
        ACTIVE = 3
        FAILED = 4
    createTime = _messages.StringField(1)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 2)
    expireTime = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)