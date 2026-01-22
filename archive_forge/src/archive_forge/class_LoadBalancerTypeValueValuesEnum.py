from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LoadBalancerTypeValueValuesEnum(_messages.Enum):
    """The type of load balancer specified by this target. This value must
    match the configuration of the load balancer located at the
    LoadBalancerTarget's IP address, port, and region. Use the following: -
    *regionalL4ilb*: for a regional internal passthrough Network Load
    Balancer. - *regionalL7ilb*: for a regional internal Application Load
    Balancer. - *globalL7ilb*: for a global internal Application Load
    Balancer.

    Values:
      none: <no description>
      globalL7ilb: <no description>
      regionalL4ilb: <no description>
      regionalL7ilb: <no description>
    """
    none = 0
    globalL7ilb = 1
    regionalL4ilb = 2
    regionalL7ilb = 3