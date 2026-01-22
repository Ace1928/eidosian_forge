from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupVault(_messages.Message):
    """Message describing BackupVault object.

  Enums:
    StateValueValuesEnum: Output only. The BackupVault resource instance
      state.

  Messages:
    LabelsValue: Optional. Resource labels to represent user provided
      metadata. No labels currently defined:

  Fields:
    backupCount: Output only. The number of backups in this backup vault.
    createTime: Output only. The time when the instance was created.
    deletable: Output only. Set to true when there are no backups nested under
      this resource.
    description: Optional. The description of the BackupVault instance (2048
      characters or less).
    effectiveTime: Optional. Time after which the BackupVault resource is
      locked.
    enforcedRetentionDuration: Required. The default retention period for each
      backup in the backup vault.
    etag: Optional. Server specified ETag for the backup vault resource to
      prevent simultaneous updates from overwiting each other.
    labels: Optional. Resource labels to represent user provided metadata. No
      labels currently defined:
    name: Output only. The resource name.
    serviceAccount: Output only. Service account used by the BackupVault
      Service for this BackupVault. The user should grant this account
      permissions in their workload project to enable the service to run
      backups and restores there.
    state: Output only. The BackupVault resource instance state.
    totalStoredBytes: Output only. Total size of the storage used by all
      backup resources.
    updateTime: Output only. The time when the instance was updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The BackupVault resource instance state.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: The backup vault is being created.
      ACTIVE: The backup vault has been created and is fully usable.
      DELETING: The backup vault is being deleted.
      ERROR: The backup vault is experiencing an issue and might be unusable.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        ERROR = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user provided metadata. No
    labels currently defined:

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    backupCount = _messages.IntegerField(1)
    createTime = _messages.StringField(2)
    deletable = _messages.BooleanField(3)
    description = _messages.StringField(4)
    effectiveTime = _messages.StringField(5)
    enforcedRetentionDuration = _messages.StringField(6)
    etag = _messages.StringField(7)
    labels = _messages.MessageField('LabelsValue', 8)
    name = _messages.StringField(9)
    serviceAccount = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    totalStoredBytes = _messages.IntegerField(12)
    updateTime = _messages.StringField(13)