from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FraudPreventionValueValuesEnum(_messages.Enum):
    """Optional. The Fraud Prevention setting for this assessment.

    Values:
      FRAUD_PREVENTION_UNSPECIFIED: Default, unspecified setting. If opted in
        for automatic detection, `fraud_prevention_assessment` is returned
        based on the request. Otherwise, `fraud_prevention_assessment` is
        returned if `transaction_data` is present in the `Event` and Fraud
        Prevention is enabled in the Google Cloud console.
      ENABLED: Enable Fraud Prevention for this assessment, if Fraud
        Prevention is enabled in the Google Cloud console.
      DISABLED: Disable Fraud Prevention for this assessment, regardless of
        opt-in status or Google Cloud console settings.
    """
    FRAUD_PREVENTION_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2