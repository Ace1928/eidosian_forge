from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationAttemptInfo(_messages.Message):
    """State of the latest attempt to authorize a domain for certificate
  issuance.

  Enums:
    FailureReasonValueValuesEnum: Output only. Reason for failure of the
      authorization attempt for the domain.
    StateValueValuesEnum: Output only. State of the domain for managed
      certificate issuance.

  Fields:
    details: Output only. Human readable explanation for reaching the state.
      Provided to help address the configuration issues. Not guaranteed to be
      stable. For programmatic access use FailureReason enum.
    domain: Domain name of the authorization attempt.
    failureReason: Output only. Reason for failure of the authorization
      attempt for the domain.
    state: Output only. State of the domain for managed certificate issuance.
  """

    class FailureReasonValueValuesEnum(_messages.Enum):
        """Output only. Reason for failure of the authorization attempt for the
    domain.

    Values:
      FAILURE_REASON_UNSPECIFIED: <no description>
      CONFIG: There was a problem with the user's DNS or load balancer
        configuration for this domain.
      CAA: Certificate issuance forbidden by an explicit CAA record for the
        domain or a failure to check CAA records for the domain.
      RATE_LIMITED: Reached a CA or internal rate-limit for the domain, e.g.
        for certificates per top-level private domain.
    """
        FAILURE_REASON_UNSPECIFIED = 0
        CONFIG = 1
        CAA = 2
        RATE_LIMITED = 3

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the domain for managed certificate issuance.

    Values:
      STATE_UNSPECIFIED: <no description>
      AUTHORIZING: Certificate provisioning for this domain is under way. GCP
        will attempt to authorize the domain.
      AUTHORIZED: A managed certificate can be provisioned, no issues for this
        domain.
      FAILED: Attempt to authorize the domain failed. This prevents the
        Managed Certificate from being issued. See `failure_reason` and
        `details` fields for more information.
    """
        STATE_UNSPECIFIED = 0
        AUTHORIZING = 1
        AUTHORIZED = 2
        FAILED = 3
    details = _messages.StringField(1)
    domain = _messages.StringField(2)
    failureReason = _messages.EnumField('FailureReasonValueValuesEnum', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)