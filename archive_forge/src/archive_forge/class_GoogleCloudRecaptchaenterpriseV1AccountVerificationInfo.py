from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1AccountVerificationInfo(_messages.Message):
    """Information about account verification, used for identity verification.

  Enums:
    LatestVerificationResultValueValuesEnum: Output only. Result of the latest
      account verification challenge.

  Fields:
    endpoints: Optional. Endpoints that can be used for identity verification.
    languageCode: Optional. Language code preference for the verification
      message, set as a IETF BCP 47 language code.
    latestVerificationResult: Output only. Result of the latest account
      verification challenge.
    username: Username of the account that is being verified. Deprecated.
      Customers should now provide the `account_id` field in
      `event.user_info`.
  """

    class LatestVerificationResultValueValuesEnum(_messages.Enum):
        """Output only. Result of the latest account verification challenge.

    Values:
      RESULT_UNSPECIFIED: No information about the latest account
        verification.
      SUCCESS_USER_VERIFIED: The user was successfully verified. This means
        the account verification challenge was successfully completed.
      ERROR_USER_NOT_VERIFIED: The user failed the verification challenge.
      ERROR_SITE_ONBOARDING_INCOMPLETE: The site is not properly onboarded to
        use the account verification feature.
      ERROR_RECIPIENT_NOT_ALLOWED: The recipient is not allowed for account
        verification. This can occur during integration but should not occur
        in production.
      ERROR_RECIPIENT_ABUSE_LIMIT_EXHAUSTED: The recipient has already been
        sent too many verification codes in a short amount of time.
      ERROR_CRITICAL_INTERNAL: The verification flow could not be completed
        due to a critical internal error.
      ERROR_CUSTOMER_QUOTA_EXHAUSTED: The client has exceeded their two factor
        request quota for this period of time.
      ERROR_VERIFICATION_BYPASSED: The request cannot be processed at the time
        because of an incident. This bypass can be restricted to a problematic
        destination email domain, a customer, or could affect the entire
        service.
      ERROR_VERDICT_MISMATCH: The request parameters do not match with the
        token provided and cannot be processed.
    """
        RESULT_UNSPECIFIED = 0
        SUCCESS_USER_VERIFIED = 1
        ERROR_USER_NOT_VERIFIED = 2
        ERROR_SITE_ONBOARDING_INCOMPLETE = 3
        ERROR_RECIPIENT_NOT_ALLOWED = 4
        ERROR_RECIPIENT_ABUSE_LIMIT_EXHAUSTED = 5
        ERROR_CRITICAL_INTERNAL = 6
        ERROR_CUSTOMER_QUOTA_EXHAUSTED = 7
        ERROR_VERIFICATION_BYPASSED = 8
        ERROR_VERDICT_MISMATCH = 9
    endpoints = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1EndpointVerificationInfo', 1, repeated=True)
    languageCode = _messages.StringField(2)
    latestVerificationResult = _messages.EnumField('LatestVerificationResultValueValuesEnum', 3)
    username = _messages.StringField(4)