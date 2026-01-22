from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OwnershipScopeValueValuesEnum(_messages.Enum):
    """OwnershipScopeValueValuesEnum enum type.

    Values:
      OWNERSHIP_SCOPE_UNSPECIFIED: Unspecified ownership scope, same as
        ALL_USERS.
      ALL_USERS: Both billing account-level users and project-level users have
        full access to the budget, if the users have the required IAM
        permissions.
      BILLING_ACCOUNT: Only billing account-level users have full access to
        the budget. Project-level users have read-only access, even if they
        have the required IAM permissions.
    """
    OWNERSHIP_SCOPE_UNSPECIFIED = 0
    ALL_USERS = 1
    BILLING_ACCOUNT = 2