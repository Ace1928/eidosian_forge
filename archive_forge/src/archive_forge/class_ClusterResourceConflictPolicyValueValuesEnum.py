from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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