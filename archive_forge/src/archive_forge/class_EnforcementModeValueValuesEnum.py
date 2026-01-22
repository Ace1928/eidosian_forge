from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnforcementModeValueValuesEnum(_messages.Enum):
    """Required. The action when a pod creation is denied by the admission
    rule.

    Values:
      ENFORCEMENT_MODE_UNSPECIFIED: Do not use.
      ENFORCED_BLOCK_AND_AUDIT_LOG: Enforce the admission rule by blocking the
        pod creation.
      DRYRUN_AUDIT_LOG_ONLY: Dryrun mode: Audit logging only. This will allow
        the pod creation as if the admission request had specified break-
        glass.
    """
    ENFORCEMENT_MODE_UNSPECIFIED = 0
    ENFORCED_BLOCK_AND_AUDIT_LOG = 1
    DRYRUN_AUDIT_LOG_ONLY = 2