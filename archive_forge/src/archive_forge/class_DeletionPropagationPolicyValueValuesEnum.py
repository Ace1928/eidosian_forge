from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeletionPropagationPolicyValueValuesEnum(_messages.Enum):
    """Deletion propagation policy of the rollout.

    Values:
      DELETION_PROPAGATION_POLICY_UNSPECIFIED: Unspecified deletion
        propagation policy. Defaults to FOREGROUND.
      FOREGROUND: Foreground deletion propagation policy. Any resources synced
        to the cluster will be deleted.
      ORPHAN: Orphan deletion propagation policy. Any resources synced to the
        cluster will be abandoned.
    """
    DELETION_PROPAGATION_POLICY_UNSPECIFIED = 0
    FOREGROUND = 1
    ORPHAN = 2