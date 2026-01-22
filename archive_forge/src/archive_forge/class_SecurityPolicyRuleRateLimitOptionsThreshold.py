from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleRateLimitOptionsThreshold(_messages.Message):
    """A SecurityPolicyRuleRateLimitOptionsThreshold object.

  Fields:
    count: Number of HTTP(S) requests for calculating the threshold.
    intervalSec: Interval over which the threshold is computed.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    intervalSec = _messages.IntegerField(2, variant=_messages.Variant.INT32)