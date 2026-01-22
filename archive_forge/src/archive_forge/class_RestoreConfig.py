from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreConfig(_messages.Message):
    """Configuration of a restore.

  Enums:
    ClusterResourceConflictPolicyValueValuesEnum: Optional. Defines the
      behavior for handling the situation where cluster-scoped resources being
      restored already exist in the target cluster. This MUST be set to a
      value other than CLUSTER_RESOURCE_CONFLICT_POLICY_UNSPECIFIED if
      cluster_resource_restore_scope is not empty.
    NamespacedResourceRestoreModeValueValuesEnum: Optional. Defines the
      behavior for handling the situation where sets of namespaced resources
      being restored already exist in the target cluster. This MUST be set to
      a value other than NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED.
    VolumeDataRestorePolicyValueValuesEnum: Optional. Specifies the mechanism
      to be used to restore volume data. Default:
      VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED (will be treated as
      NO_VOLUME_DATA_RESTORATION).

  Fields:
    allNamespaces: Restore all namespaced resources in the Backup if set to
      "True". Specifying this field to "False" is an error.
    clusterResourceConflictPolicy: Optional. Defines the behavior for handling
      the situation where cluster-scoped resources being restored already
      exist in the target cluster. This MUST be set to a value other than
      CLUSTER_RESOURCE_CONFLICT_POLICY_UNSPECIFIED if
      cluster_resource_restore_scope is not empty.
    clusterResourceRestoreScope: Optional. Identifies the cluster-scoped
      resources to restore from the Backup. Not specifying it means NO cluster
      resource will be restored.
    excludedNamespaces: A list of selected namespaces excluded from
      restoration. All namespaces except those in this list will be restored.
    namespacedResourceRestoreMode: Optional. Defines the behavior for handling
      the situation where sets of namespaced resources being restored already
      exist in the target cluster. This MUST be set to a value other than
      NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED.
    noNamespaces: Do not restore any namespaced resources if set to "True".
      Specifying this field to "False" is not allowed.
    restoreOrder: Optional. RestoreOrder contains custom ordering to use on a
      Restore.
    selectedApplications: A list of selected ProtectedApplications to restore.
      The listed ProtectedApplications and all the resources to which they
      refer will be restored.
    selectedNamespaces: A list of selected Namespaces to restore from the
      Backup. The listed Namespaces and all resources contained in them will
      be restored.
    substitutionRules: Optional. A list of transformation rules to be applied
      against Kubernetes resources as they are selected for restoration from a
      Backup. Rules are executed in order defined - this order matters, as
      changes made by a rule may impact the filtering logic of subsequent
      rules. An empty list means no substitution will occur.
    transformationRules: Optional. A list of transformation rules to be
      applied against Kubernetes resources as they are selected for
      restoration from a Backup. Rules are executed in order defined - this
      order matters, as changes made by a rule may impact the filtering logic
      of subsequent rules. An empty list means no transformation will occur.
    volumeDataRestorePolicy: Optional. Specifies the mechanism to be used to
      restore volume data. Default: VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED
      (will be treated as NO_VOLUME_DATA_RESTORATION).
    volumeDataRestorePolicyBindings: Optional. A table that binds volumes by
      their scope to a restore policy. Bindings must have a unique scope. Any
      volumes not scoped in the bindings are subject to the policy defined in
      volume_data_restore_policy.
  """

    class ClusterResourceConflictPolicyValueValuesEnum(_messages.Enum):
        """Optional. Defines the behavior for handling the situation where
    cluster-scoped resources being restored already exist in the target
    cluster. This MUST be set to a value other than
    CLUSTER_RESOURCE_CONFLICT_POLICY_UNSPECIFIED if
    cluster_resource_restore_scope is not empty.

    Values:
      CLUSTER_RESOURCE_CONFLICT_POLICY_UNSPECIFIED: Unspecified. Only allowed
        if no cluster-scoped resources will be restored.
      USE_EXISTING_VERSION: Do not attempt to restore the conflicting
        resource.
      USE_BACKUP_VERSION: Delete the existing version before re-creating it
        from the Backup. This is a dangerous option which could cause
        unintentional data loss if used inappropriately. For example, deleting
        a CRD will cause Kubernetes to delete all CRs of that type.
    """
        CLUSTER_RESOURCE_CONFLICT_POLICY_UNSPECIFIED = 0
        USE_EXISTING_VERSION = 1
        USE_BACKUP_VERSION = 2

    class NamespacedResourceRestoreModeValueValuesEnum(_messages.Enum):
        """Optional. Defines the behavior for handling the situation where sets
    of namespaced resources being restored already exist in the target
    cluster. This MUST be set to a value other than
    NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED.

    Values:
      NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED: Unspecified (invalid).
      DELETE_AND_RESTORE: When conflicting top-level resources (either
        Namespaces or ProtectedApplications, depending upon the scope) are
        encountered, this will first trigger a delete of the conflicting
        resource AND ALL OF ITS REFERENCED RESOURCES (e.g., all resources in
        the Namespace or all resources referenced by the ProtectedApplication)
        before restoring the resources from the Backup. This mode should only
        be used when you are intending to revert some portion of a cluster to
        an earlier state.
      FAIL_ON_CONFLICT: If conflicting top-level resources (either Namespaces
        or ProtectedApplications, depending upon the scope) are encountered at
        the beginning of a restore process, the Restore will fail. If a
        conflict occurs during the restore process itself (e.g., because an
        out of band process creates conflicting resources), a conflict will be
        reported.
      MERGE_SKIP_ON_CONFLICT: This mode merges the backup and the target
        cluster and skips the conflicting resources. If a single resource to
        restore exists in the cluster before restoration, the resource will be
        skipped, otherwise it will be restored.
      MERGE_REPLACE_VOLUME_ON_CONFLICT: This mode merges the backup and the
        target cluster and skips the conflicting resources except volume data.
        If a PVC to restore already exists, this mode will restore/reconnect
        the volume without overwriting the PVC. It is similar to
        MERGE_SKIP_ON_CONFLICT except that it will apply the volume data
        policy for the conflicting PVCs: - RESTORE_VOLUME_DATA_FROM_BACKUP:
        restore data only and respect the reclaim policy of the original PV; -
        REUSE_VOLUME_HANDLE_FROM_BACKUP: reconnect and respect the reclaim
        policy of the original PV; - NO_VOLUME_DATA_RESTORATION: new provision
        and respect the reclaim policy of the original PV. Note that this mode
        could cause data loss as the original PV can be retained or deleted
        depending on its reclaim policy.
      MERGE_REPLACE_ON_CONFLICT: This mode merges the backup and the target
        cluster and replaces the conflicting resources with the ones in the
        backup. If a single resource to restore exists in the cluster before
        restoration, the resource will be replaced with the one from the
        backup. To replace an existing resource, the first attempt is to
        update the resource to match the one from the backup; if the update
        fails, the second attempt is to delete the resource and restore it
        from the backup. Note that this mode could cause data loss as it
        replaces the existing resources in the target cluster, and the
        original PV can be retained or deleted depending on its reclaim
        policy.
    """
        NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED = 0
        DELETE_AND_RESTORE = 1
        FAIL_ON_CONFLICT = 2
        MERGE_SKIP_ON_CONFLICT = 3
        MERGE_REPLACE_VOLUME_ON_CONFLICT = 4
        MERGE_REPLACE_ON_CONFLICT = 5

    class VolumeDataRestorePolicyValueValuesEnum(_messages.Enum):
        """Optional. Specifies the mechanism to be used to restore volume data.
    Default: VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED (will be treated as
    NO_VOLUME_DATA_RESTORATION).

    Values:
      VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED: Unspecified (illegal).
      RESTORE_VOLUME_DATA_FROM_BACKUP: For each PVC to be restored, create a
        new underlying volume and PV from the corresponding VolumeBackup
        contained within the Backup.
      REUSE_VOLUME_HANDLE_FROM_BACKUP: For each PVC to be restored, attempt to
        reuse the original PV contained in the Backup (with its original
        underlying volume). This option is likely only usable when restoring a
        workload to its original cluster.
      NO_VOLUME_DATA_RESTORATION: For each PVC to be restored, create PVC
        without any particular action to restore data. In this case, the
        normal Kubernetes provisioning logic would kick in, and this would
        likely result in either dynamically provisioning blank PVs or binding
        to statically provisioned PVs.
    """
        VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED = 0
        RESTORE_VOLUME_DATA_FROM_BACKUP = 1
        REUSE_VOLUME_HANDLE_FROM_BACKUP = 2
        NO_VOLUME_DATA_RESTORATION = 3
    allNamespaces = _messages.BooleanField(1)
    clusterResourceConflictPolicy = _messages.EnumField('ClusterResourceConflictPolicyValueValuesEnum', 2)
    clusterResourceRestoreScope = _messages.MessageField('ClusterResourceRestoreScope', 3)
    excludedNamespaces = _messages.MessageField('Namespaces', 4)
    namespacedResourceRestoreMode = _messages.EnumField('NamespacedResourceRestoreModeValueValuesEnum', 5)
    noNamespaces = _messages.BooleanField(6)
    restoreOrder = _messages.MessageField('RestoreOrder', 7)
    selectedApplications = _messages.MessageField('NamespacedNames', 8)
    selectedNamespaces = _messages.MessageField('Namespaces', 9)
    substitutionRules = _messages.MessageField('SubstitutionRule', 10, repeated=True)
    transformationRules = _messages.MessageField('TransformationRule', 11, repeated=True)
    volumeDataRestorePolicy = _messages.EnumField('VolumeDataRestorePolicyValueValuesEnum', 12)
    volumeDataRestorePolicyBindings = _messages.MessageField('VolumeDataRestorePolicyBinding', 13, repeated=True)