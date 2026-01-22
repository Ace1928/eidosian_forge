from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class VerificationStateValueValuesEnum(_messages.Enum):
    """The verification state of this contact's email address.

    Values:
      VERIFICATION_STATE_UNSPECIFIED: VerificationState is unrecognized or
        unspecified.
      PENDING: Verification was sent but has not been accepted yet. A contact
        will remain in this state even if verification time limit elapses. At
        that point the contact cannot be verified, and ResendVerification must
        be called to reset the timeout.
      VERIFIED: Email has been verified.
      FAILED: Error with verification - email could not be delivered.
    """
    VERIFICATION_STATE_UNSPECIFIED = 0
    PENDING = 1
    VERIFIED = 2
    FAILED = 3