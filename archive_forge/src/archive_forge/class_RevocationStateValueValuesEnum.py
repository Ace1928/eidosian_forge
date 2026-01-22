from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevocationStateValueValuesEnum(_messages.Enum):
    """Indicates why a Certificate was revoked.

    Values:
      REVOCATION_REASON_UNSPECIFIED: Default unspecified value. This value
        does indicate that a Certificate has been revoked, but that a reason
        has not been recorded.
      KEY_COMPROMISE: Key material for this Certificate may have leaked.
      CERTIFICATE_AUTHORITY_COMPROMISE: The key material for a certificate
        authority in the issuing path may have leaked.
      AFFILIATION_CHANGED: The subject or other attributes in this Certificate
        have changed.
      SUPERSEDED: This Certificate has been superseded.
      CESSATION_OF_OPERATION: This Certificate or entities in the issuing path
        have ceased to operate.
      CERTIFICATE_HOLD: This Certificate should not be considered valid, it is
        expected that it may become valid in the future.
      PRIVILEGE_WITHDRAWN: This Certificate no longer has permission to assert
        the listed attributes.
      ATTRIBUTE_AUTHORITY_COMPROMISE: The authority which determines
        appropriate attributes for a Certificate may have been compromised.
    """
    REVOCATION_REASON_UNSPECIFIED = 0
    KEY_COMPROMISE = 1
    CERTIFICATE_AUTHORITY_COMPROMISE = 2
    AFFILIATION_CHANGED = 3
    SUPERSEDED = 4
    CESSATION_OF_OPERATION = 5
    CERTIFICATE_HOLD = 6
    PRIVILEGE_WITHDRAWN = 7
    ATTRIBUTE_AUTHORITY_COMPROMISE = 8