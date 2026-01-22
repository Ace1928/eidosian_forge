from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingInteractionValueValuesEnum(_messages.Enum):
    """BillingInteractionValueValuesEnum enum type.

    Values:
      BILLING_INTERACTION_UNSPECIFIED: The interaction between this quota
        group and the project billing status is unspecified.
      NONBILLABLE_ONLY: This quota group is enforced only when the consumer
        project is not billable.
      BILLABLE_ONLY: This quota group is enforced only when the consumer
        project is billable.
      ANY_BILLING_STATUS: This quota group is enforced regardless of the
        consumer project's billing status.
    """
    BILLING_INTERACTION_UNSPECIFIED = 0
    NONBILLABLE_ONLY = 1
    BILLABLE_ONLY = 2
    ANY_BILLING_STATUS = 3