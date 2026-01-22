from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TokenProperties(_messages.Message):
    """Properties of the provided event token.

  Enums:
    InvalidReasonValueValuesEnum: Output only. Reason associated with the
      response when valid = false.

  Fields:
    action: Output only. Action name provided at token generation.
    androidPackageName: Output only. The name of the Android package with
      which the token was generated (Android keys only).
    createTime: Output only. The timestamp corresponding to the generation of
      the token.
    hostname: Output only. The hostname of the page on which the token was
      generated (Web keys only).
    invalidReason: Output only. Reason associated with the response when valid
      = false.
    iosBundleId: Output only. The ID of the iOS bundle with which the token
      was generated (iOS keys only).
    valid: Output only. Whether the provided user response token is valid.
      When valid = false, the reason could be specified in invalid_reason or
      it could also be due to a user failing to solve a challenge or a sitekey
      mismatch (i.e the sitekey used to generate the token was different than
      the one specified in the assessment).
  """

    class InvalidReasonValueValuesEnum(_messages.Enum):
        """Output only. Reason associated with the response when valid = false.

    Values:
      INVALID_REASON_UNSPECIFIED: Default unspecified type.
      UNKNOWN_INVALID_REASON: If the failure reason was not accounted for.
      MALFORMED: The provided user verification token was malformed.
      EXPIRED: The user verification token had expired.
      DUPE: The user verification had already been seen.
      MISSING: The user verification token was not present.
      BROWSER_ERROR: A retriable error (such as network failure) occurred on
        the browser. Could easily be simulated by an attacker.
    """
        INVALID_REASON_UNSPECIFIED = 0
        UNKNOWN_INVALID_REASON = 1
        MALFORMED = 2
        EXPIRED = 3
        DUPE = 4
        MISSING = 5
        BROWSER_ERROR = 6
    action = _messages.StringField(1)
    androidPackageName = _messages.StringField(2)
    createTime = _messages.StringField(3)
    hostname = _messages.StringField(4)
    invalidReason = _messages.EnumField('InvalidReasonValueValuesEnum', 5)
    iosBundleId = _messages.StringField(6)
    valid = _messages.BooleanField(7)