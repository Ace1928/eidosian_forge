from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnforcementActionValueValuesEnum(_messages.Enum):
    """spec.enforcement_action.

    Values:
      ENFORCEMENT_ACTION_UNSPECIFIED: Unspecified state for an enforcement
        action.
      ENFORCEMENT_ACTION_DENY: The resource is denied admission to the
        membership.
      ENFORCEMENT_ACTION_DRYRUN: Allows testing constraints without enforcing
        them.
      ENFORCEMENT_ACTION_WARN: Provides immediate feedback on why a resource
        violates a constraint.
    """
    ENFORCEMENT_ACTION_UNSPECIFIED = 0
    ENFORCEMENT_ACTION_DENY = 1
    ENFORCEMENT_ACTION_DRYRUN = 2
    ENFORCEMENT_ACTION_WARN = 3