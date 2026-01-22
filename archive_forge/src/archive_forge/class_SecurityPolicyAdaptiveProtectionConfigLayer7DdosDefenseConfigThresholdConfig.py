from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfig(_messages.Message):
    """A
  SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfig
  object.

  Fields:
    autoDeployConfidenceThreshold: A number attribute.
    autoDeployExpirationSec: A integer attribute.
    autoDeployImpactedBaselineThreshold: A number attribute.
    autoDeployLoadThreshold: A number attribute.
    detectionAbsoluteQps: A number attribute.
    detectionLoadThreshold: A number attribute.
    detectionRelativeToBaselineQps: A number attribute.
    name: The name must be 1-63 characters long, and comply with RFC1035. The
      name must be unique within the security policy.
    trafficGranularityConfigs: Configuration options for enabling Adaptive
      Protection to operate on specified granular traffic units.
  """
    autoDeployConfidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    autoDeployExpirationSec = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    autoDeployImpactedBaselineThreshold = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    autoDeployLoadThreshold = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    detectionAbsoluteQps = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    detectionLoadThreshold = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    detectionRelativeToBaselineQps = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    name = _messages.StringField(8)
    trafficGranularityConfigs = _messages.MessageField('SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfigTrafficGranularityConfig', 9, repeated=True)