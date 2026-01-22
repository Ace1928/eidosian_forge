from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectApplicationAwareInterconnect(_messages.Message):
    """Configuration information for enabling Application Aware Interconnect
  (AAI) on this Cloud Interconnect connection between Google and your on-
  premises router.

  Fields:
    bandwidthPercentagePolicy: A
      InterconnectApplicationAwareInterconnectBandwidthPercentagePolicy
      attribute.
    profileDescription: A description for the AAI profile on this
      interconnect.
    strictPriorityPolicy: A
      InterconnectApplicationAwareInterconnectStrictPriorityPolicy attribute.
  """
    bandwidthPercentagePolicy = _messages.MessageField('InterconnectApplicationAwareInterconnectBandwidthPercentagePolicy', 1)
    profileDescription = _messages.StringField(2)
    strictPriorityPolicy = _messages.MessageField('InterconnectApplicationAwareInterconnectStrictPriorityPolicy', 3)