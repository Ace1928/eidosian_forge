from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeBackup(_messages.Message):
    """Represents the backup of a specific persistent volume as a component of
  a Backup - both the record of the operation and a pointer to the underlying
  storage-specific artifacts.

  Enums:
    FormatValueValuesEnum: Output only. The format used for the volume backup.
    StateValueValuesEnum: Output only. The current state of this VolumeBackup.

  Fields:
    completeTime: Output only. The timestamp when the associated underlying
      volume backup operation completed.
    createTime: Output only. The timestamp when this VolumeBackup resource was
      created.
    diskSizeBytes: Output only. The minimum size of the disk to which this
      VolumeBackup can be restored.
    etag: Output only. `etag` is used for optimistic concurrency control as a
      way to help prevent simultaneous updates of a volume backup from
      overwriting each other. It is strongly suggested that systems make use
      of the `etag` in the read-modify-write cycle to perform volume backup
      updates in order to avoid race conditions.
    format: Output only. The format used for the volume backup.
    name: Output only. The full name of the VolumeBackup resource. Format:
      `projects/*/locations/*/backupPlans/*/backups/*/volumeBackups/*`.
    sourcePvc: Output only. A reference to the source Kubernetes PVC from
      which this VolumeBackup was created.
    state: Output only. The current state of this VolumeBackup.
    stateMessage: Output only. A human readable message explaining why the
      VolumeBackup is in its current state.
    storageBytes: Output only. The aggregate size of the underlying artifacts
      associated with this VolumeBackup in the backup storage. This may change
      over time when multiple backups of the same volume share the same backup
      storage location. In particular, this is likely to increase in size when
      the immediately preceding backup of the same volume is deleted.
    uid: Output only. Server generated global unique identifier of
      [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier)
      format.
    updateTime: Output only. The timestamp when this VolumeBackup resource was
      last updated.
    volumeBackupHandle: Output only. A storage system-specific opaque handle
      to the underlying volume backup.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Output only. The format used for the volume backup.

    Values:
      VOLUME_BACKUP_FORMAT_UNSPECIFIED: Default value, not specified.
      GCE_PERSISTENT_DISK: Compute Engine Persistent Disk snapshot based
        volume backup.
    """
        VOLUME_BACKUP_FORMAT_UNSPECIFIED = 0
        GCE_PERSISTENT_DISK = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of this VolumeBackup.

    Values:
      STATE_UNSPECIFIED: This is an illegal state and should not be
        encountered.
      CREATING: A volume for the backup was identified and backup process is
        about to start.
      SNAPSHOTTING: The volume backup operation has begun and is in the
        initial "snapshot" phase of the process. Any defined
        ProtectedApplication "pre" hooks will be executed before entering this
        state and "post" hooks will be executed upon leaving this state.
      UPLOADING: The snapshot phase of the volume backup operation has
        completed and the snapshot is now being uploaded to backup storage.
      SUCCEEDED: The volume backup operation has completed successfully.
      FAILED: The volume backup operation has failed.
      DELETING: This VolumeBackup resource (and its associated artifacts) is
        in the process of being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        SNAPSHOTTING = 2
        UPLOADING = 3
        SUCCEEDED = 4
        FAILED = 5
        DELETING = 6
    completeTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    diskSizeBytes = _messages.IntegerField(3)
    etag = _messages.StringField(4)
    format = _messages.EnumField('FormatValueValuesEnum', 5)
    name = _messages.StringField(6)
    sourcePvc = _messages.MessageField('NamespacedName', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    stateMessage = _messages.StringField(9)
    storageBytes = _messages.IntegerField(10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)
    volumeBackupHandle = _messages.StringField(13)