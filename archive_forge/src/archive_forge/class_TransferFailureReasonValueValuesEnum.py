from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferFailureReasonValueValuesEnum(_messages.Enum):
    """Output only. Deprecated: For more information, see [Cloud Domains
    feature
    deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
    deprecations). The reason the domain transfer failed. Only set for domains
    in TRANSFER_FAILED state.

    Values:
      TRANSFER_FAILURE_REASON_UNSPECIFIED: Transfer failure unspecified.
      TRANSFER_FAILURE_REASON_UNKNOWN: Transfer failed for an unknown reason.
      EMAIL_CONFIRMATION_FAILURE: An email confirmation sent to the user was
        rejected or expired.
      DOMAIN_NOT_REGISTERED: The domain is available for registration.
      DOMAIN_HAS_TRANSFER_LOCK: The domain has a transfer lock with its
        current registrar which must be removed prior to transfer.
      INVALID_AUTHORIZATION_CODE: The authorization code entered is not valid.
      TRANSFER_CANCELLED: The transfer was cancelled by the domain owner,
        current registrar, or TLD registry.
      TRANSFER_REJECTED: The transfer was rejected by the current registrar.
        Contact the current registrar for more information.
      INVALID_REGISTRANT_EMAIL_ADDRESS: The registrant email address cannot be
        parsed from the domain's current public contact data.
      DOMAIN_NOT_ELIGIBLE_FOR_TRANSFER: The domain is not eligible for
        transfer due requirements imposed by the current registrar or TLD
        registry.
      TRANSFER_ALREADY_PENDING: Another transfer is already pending for this
        domain. The existing transfer attempt must expire or be cancelled in
        order to proceed.
    """
    TRANSFER_FAILURE_REASON_UNSPECIFIED = 0
    TRANSFER_FAILURE_REASON_UNKNOWN = 1
    EMAIL_CONFIRMATION_FAILURE = 2
    DOMAIN_NOT_REGISTERED = 3
    DOMAIN_HAS_TRANSFER_LOCK = 4
    INVALID_AUTHORIZATION_CODE = 5
    TRANSFER_CANCELLED = 6
    TRANSFER_REJECTED = 7
    INVALID_REGISTRANT_EMAIL_ADDRESS = 8
    DOMAIN_NOT_ELIGIBLE_FOR_TRANSFER = 9
    TRANSFER_ALREADY_PENDING = 10