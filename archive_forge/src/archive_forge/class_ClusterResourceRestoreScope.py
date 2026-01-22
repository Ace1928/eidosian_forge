from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterResourceRestoreScope(_messages.Message):
    """Defines the scope of cluster-scoped resources to restore. Some group
  kinds are not reasonable choices for a restore, and will cause an error if
  selected here. Any scope selection that would restore "all valid" resources
  automatically excludes these group kinds. - gkebackup.gke.io/BackupJob -
  gkebackup.gke.io/RestoreJob - metrics.k8s.io/NodeMetrics -
  migration.k8s.io/StorageState - migration.k8s.io/StorageVersionMigration -
  Node - snapshot.storage.k8s.io/VolumeSnapshotContent -
  storage.k8s.io/CSINode Some group kinds are driven by restore configuration
  elsewhere, and will cause an error if selected here. - Namespace -
  PersistentVolume

  Fields:
    allGroupKinds: Optional. If True, all valid cluster-scoped resources will
      be restored. Mutually exclusive to any other field in the message.
    excludedGroupKinds: Optional. A list of cluster-scoped resource group
      kinds to NOT restore from the backup. If specified, all valid cluster-
      scoped resources will be restored except for those specified in the
      list. Mutually exclusive to any other field in the message.
    noGroupKinds: Optional. If True, no cluster-scoped resources will be
      restored. This has the same restore scope as if the message is not
      defined. Mutually exclusive to any other field in the message.
    selectedGroupKinds: Optional. A list of cluster-scoped resource group
      kinds to restore from the backup. If specified, only the selected
      resources will be restored. Mutually exclusive to any other field in the
      message.
  """
    allGroupKinds = _messages.BooleanField(1)
    excludedGroupKinds = _messages.MessageField('GroupKind', 2, repeated=True)
    noGroupKinds = _messages.BooleanField(3)
    selectedGroupKinds = _messages.MessageField('GroupKind', 4, repeated=True)