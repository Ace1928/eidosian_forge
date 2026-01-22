from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientValidationModeValueValuesEnum(_messages.Enum):
    """When the client presents an invalid certificate or no certificate to
    the load balancer, the `client_validation_mode` specifies how the client
    connection is handled. Required if the policy is to be used with the
    external HTTPS load balancing. For Traffic Director it must be empty.

    Values:
      CLIENT_VALIDATION_MODE_UNSPECIFIED: Not allowed.
      ALLOW_INVALID_OR_MISSING_CLIENT_CERT: Allow connection even if
        certificate chain validation of the client certificate failed or no
        client certificate was presented. The proof of possession of the
        private key is always checked if client certificate was presented.
        This mode requires the backend to implement processing of data
        extracted from a client certificate to authenticate the peer, or to
        reject connections if the client certificate fingerprint is missing.
      REJECT_INVALID: Require a client certificate and allow connection to the
        backend only if validation of the client certificate passed. If set,
        requires a reference to non-empty TrustConfig specified in
        `client_validation_trust_config`.
    """
    CLIENT_VALIDATION_MODE_UNSPECIFIED = 0
    ALLOW_INVALID_OR_MISSING_CLIENT_CERT = 1
    REJECT_INVALID = 2