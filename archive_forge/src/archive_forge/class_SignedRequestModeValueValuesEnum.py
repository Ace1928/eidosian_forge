from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SignedRequestModeValueValuesEnum(_messages.Enum):
    """Optional. Specifies whether to enforce signed requests. The default
    value is DISABLED, which means all content is public, and does not
    authorize access. You must also set a signed_request_keyset to enable
    signed requests. When set to REQUIRE_SIGNATURES or REQUIRE_TOKENS, all
    matching requests get their signature validated. Requests that aren't
    signed with the corresponding private key, or that are otherwise invalid
    (such as expired or do not match the signature, IP address, or header) are
    rejected with an HTTP 403 error. If logging is turned on, then invalid
    requests are also logged.

    Values:
      SIGNED_REQUEST_MODE_UNSPECIFIED: Unspecified value. Defaults to
        `DISABLED`.
      DISABLED: Do not enforce signed requests.
      REQUIRE_SIGNATURES: Enforce signed requests using query parameter, path
        component, or cookie signatures. All requests must have a valid
        signature. Requests that are missing the signature (URL or cookie-
        based) are rejected as if the signature was invalid.
      REQUIRE_TOKENS: Enforce signed requests using signed tokens. All
        requests must have a valid signed token. Requests that are missing a
        signed token (URL or cookie-based) are rejected as if the signed token
        was invalid.
    """
    SIGNED_REQUEST_MODE_UNSPECIFIED = 0
    DISABLED = 1
    REQUIRE_SIGNATURES = 2
    REQUIRE_TOKENS = 3