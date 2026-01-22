from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminLoadBalancerConfig(_messages.Message):
    """BareMetalAdminLoadBalancerConfig specifies the load balancer
  configuration.

  Fields:
    manualLbConfig: Manually configured load balancers.
    portConfig: Configures the ports that the load balancer will listen on.
    vipConfig: The VIPs used by the load balancer.
  """
    manualLbConfig = _messages.MessageField('BareMetalAdminManualLbConfig', 1)
    portConfig = _messages.MessageField('BareMetalAdminPortConfig', 2)
    vipConfig = _messages.MessageField('BareMetalAdminVipConfig', 3)