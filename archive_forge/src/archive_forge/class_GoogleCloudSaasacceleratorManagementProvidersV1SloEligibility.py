from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1SloEligibility(_messages.Message):
    """SloEligibility is a tuple containing eligibility value: true if an
  instance is eligible for SLO calculation or false if it should be excluded
  from all SLO-related calculations along with a user-defined reason.

  Fields:
    eligible: Whether an instance is eligible or ineligible.
    reason: User-defined reason for the current value of instance eligibility.
      Usually, this can be directly mapped to the internal state. An empty
      reason is allowed.
  """
    eligible = _messages.BooleanField(1)
    reason = _messages.StringField(2)