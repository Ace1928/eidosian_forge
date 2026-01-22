from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingPeriodValueValuesEnum(_messages.Enum):
    """Frequency at which the customer will be billed.

    Values:
      BILLING_PERIOD_UNSPECIFIED: Billing period not specified.
      WEEKLY: Weekly billing period. **Note**: Not supported by Apigee at this
        time.
      MONTHLY: Monthly billing period.
    """
    BILLING_PERIOD_UNSPECIFIED = 0
    WEEKLY = 1
    MONTHLY = 2