from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChallengeSecurityPreferenceValueValuesEnum(_messages.Enum):
    """Optional. Settings for the frequency and difficulty at which this key
    triggers captcha challenges. This should only be specified for
    IntegrationTypes CHECKBOX and INVISIBLE.

    Values:
      CHALLENGE_SECURITY_PREFERENCE_UNSPECIFIED: Default type that indicates
        this enum hasn't been specified.
      USABILITY: Key tends to show fewer and easier challenges.
      BALANCE: Key tends to show balanced (in amount and difficulty)
        challenges.
      SECURITY: Key tends to show more and harder challenges.
    """
    CHALLENGE_SECURITY_PREFERENCE_UNSPECIFIED = 0
    USABILITY = 1
    BALANCE = 2
    SECURITY = 3