from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentCardTestingVerdict(_messages.Message):
    """Information about card testing fraud, where an adversary is testing
  fraudulently obtained cards or brute forcing their details.

  Fields:
    risk: Output only. Probability of this transaction attempt being part of a
      card testing attack. Values are from 0.0 (lowest) to 1.0 (highest).
  """
    risk = _messages.FloatField(1, variant=_messages.Variant.FLOAT)