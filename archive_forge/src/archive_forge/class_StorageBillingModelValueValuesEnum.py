from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBillingModelValueValuesEnum(_messages.Enum):
    """Optional. Updates storage_billing_model for the dataset.

    Values:
      STORAGE_BILLING_MODEL_UNSPECIFIED: Value not set.
      LOGICAL: Billing for logical bytes.
      PHYSICAL: Billing for physical bytes.
    """
    STORAGE_BILLING_MODEL_UNSPECIFIED = 0
    LOGICAL = 1
    PHYSICAL = 2