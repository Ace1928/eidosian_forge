from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudSignalsUserSignals(_messages.Message):
    """Signals describing the user involved in this transaction.

  Fields:
    activeDaysLowerBound: Output only. This user (based on email, phone, and
      other identifiers) has been seen on the internet for at least this
      number of days.
    syntheticRisk: Output only. Likelihood (from 0.0 to 1.0) this user
      includes synthetic components in their identity, such as a randomly
      generated email address, temporary phone number, or fake shipping
      address.
  """
    activeDaysLowerBound = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    syntheticRisk = _messages.FloatField(2, variant=_messages.Variant.FLOAT)