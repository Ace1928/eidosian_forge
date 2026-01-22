from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneLoadBalancerConfig(_messages.Message):
    """Specifies the load balancer configuration.

  Fields:
    bgpLbConfig: Configuration for BGP typed load balancers.
    manualLbConfig: Manually configured load balancers.
    metalLbConfig: Configuration for MetalLB load balancers.
    portConfig: Configures the ports that the load balancer will listen on.
    vipConfig: The VIPs used by the load balancer.
  """
    bgpLbConfig = _messages.MessageField('BareMetalStandaloneBgpLbConfig', 1)
    manualLbConfig = _messages.MessageField('BareMetalStandaloneManualLbConfig', 2)
    metalLbConfig = _messages.MessageField('BareMetalStandaloneMetalLbConfig', 3)
    portConfig = _messages.MessageField('BareMetalStandalonePortConfig', 4)
    vipConfig = _messages.MessageField('BareMetalStandaloneVipConfig', 5)