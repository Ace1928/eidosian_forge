from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalLoadBalancerConfig(_messages.Message):
    """Specifies the load balancer configuration.

  Fields:
    bgpLbConfig: Configuration for BGP typed load balancers. When set
      network_config.advanced_networking is automatically set to true.
    manualLbConfig: Manually configured load balancers.
    metalLbConfig: Configuration for MetalLB load balancers.
    portConfig: Configures the ports that the load balancer will listen on.
    vipConfig: The VIPs used by the load balancer.
  """
    bgpLbConfig = _messages.MessageField('BareMetalBgpLbConfig', 1)
    manualLbConfig = _messages.MessageField('BareMetalManualLbConfig', 2)
    metalLbConfig = _messages.MessageField('BareMetalMetalLbConfig', 3)
    portConfig = _messages.MessageField('BareMetalPortConfig', 4)
    vipConfig = _messages.MessageField('BareMetalVipConfig', 5)