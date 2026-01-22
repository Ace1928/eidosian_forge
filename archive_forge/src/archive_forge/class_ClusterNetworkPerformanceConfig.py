from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterNetworkPerformanceConfig(_messages.Message):
    """Configuration of network bandwidth tiers

  Enums:
    TotalEgressBandwidthTierValueValuesEnum: Specifies the total network
      bandwidth tier for NodePools in the cluster.

  Fields:
    totalEgressBandwidthTier: Specifies the total network bandwidth tier for
      NodePools in the cluster.
  """

    class TotalEgressBandwidthTierValueValuesEnum(_messages.Enum):
        """Specifies the total network bandwidth tier for NodePools in the
    cluster.

    Values:
      TIER_UNSPECIFIED: Default value
      TIER_1: Higher bandwidth, actual values based on VM size.
    """
        TIER_UNSPECIFIED = 0
        TIER_1 = 1
    totalEgressBandwidthTier = _messages.EnumField('TotalEgressBandwidthTierValueValuesEnum', 1)