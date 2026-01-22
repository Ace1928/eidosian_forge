from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminLoadBalancerConfig(_messages.Message):
    """VmwareAdminLoadBalancerConfig contains load balancer configuration for
  VMware admin cluster.

  Fields:
    f5Config: Configuration for F5 Big IP typed load balancers.
    manualLbConfig: Manually configured load balancers.
    metalLbConfig: MetalLB load balancers.
    seesawConfig: Output only. Configuration for Seesaw typed load balancers.
    vipConfig: The VIPs used by the load balancer.
  """
    f5Config = _messages.MessageField('VmwareAdminF5BigIpConfig', 1)
    manualLbConfig = _messages.MessageField('VmwareAdminManualLbConfig', 2)
    metalLbConfig = _messages.MessageField('VmwareAdminMetalLbConfig', 3)
    seesawConfig = _messages.MessageField('VmwareAdminSeesawConfig', 4)
    vipConfig = _messages.MessageField('VmwareAdminVipConfig', 5)