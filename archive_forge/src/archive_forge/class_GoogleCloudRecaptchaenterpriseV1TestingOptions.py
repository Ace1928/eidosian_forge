from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TestingOptions(_messages.Message):
    """Options for user acceptance testing.

  Enums:
    TestingChallengeValueValuesEnum: Optional. For challenge-based keys only
      (CHECKBOX, INVISIBLE), all challenge requests for this site will return
      nocaptcha if NOCAPTCHA, or an unsolvable challenge if CHALLENGE.

  Fields:
    testingChallenge: Optional. For challenge-based keys only (CHECKBOX,
      INVISIBLE), all challenge requests for this site will return nocaptcha
      if NOCAPTCHA, or an unsolvable challenge if CHALLENGE.
    testingScore: Optional. All assessments for this Key will return this
      score. Must be between 0 (likely not legitimate) and 1 (likely
      legitimate) inclusive.
  """

    class TestingChallengeValueValuesEnum(_messages.Enum):
        """Optional. For challenge-based keys only (CHECKBOX, INVISIBLE), all
    challenge requests for this site will return nocaptcha if NOCAPTCHA, or an
    unsolvable challenge if CHALLENGE.

    Values:
      TESTING_CHALLENGE_UNSPECIFIED: Perform the normal risk analysis and
        return either nocaptcha or a challenge depending on risk and trust
        factors.
      NOCAPTCHA: Challenge requests for this key always return a nocaptcha,
        which does not require a solution.
      UNSOLVABLE_CHALLENGE: Challenge requests for this key always return an
        unsolvable challenge.
    """
        TESTING_CHALLENGE_UNSPECIFIED = 0
        NOCAPTCHA = 1
        UNSOLVABLE_CHALLENGE = 2
    testingChallenge = _messages.EnumField('TestingChallengeValueValuesEnum', 1)
    testingScore = _messages.FloatField(2, variant=_messages.Variant.FLOAT)