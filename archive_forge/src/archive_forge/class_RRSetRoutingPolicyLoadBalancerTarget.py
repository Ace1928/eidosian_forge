from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyLoadBalancerTarget(_messages.Message):
    """The configuration for an individual load balancer to health check.

  Enums:
    IpProtocolValueValuesEnum: The protocol of the load balancer to health
      check.
    LoadBalancerTypeValueValuesEnum: The type of load balancer specified by
      this target. This value must match the configuration of the load
      balancer located at the LoadBalancerTarget's IP address, port, and
      region. Use the following: - *regionalL4ilb*: for a regional internal
      passthrough Network Load Balancer. - *regionalL7ilb*: for a regional
      internal Application Load Balancer. - *globalL7ilb*: for a global
      internal Application Load Balancer.

  Fields:
    ipAddress: The frontend IP address of the load balancer to health check.
    ipProtocol: The protocol of the load balancer to health check.
    kind: A string attribute.
    loadBalancerType: The type of load balancer specified by this target. This
      value must match the configuration of the load balancer located at the
      LoadBalancerTarget's IP address, port, and region. Use the following: -
      *regionalL4ilb*: for a regional internal passthrough Network Load
      Balancer. - *regionalL7ilb*: for a regional internal Application Load
      Balancer. - *globalL7ilb*: for a global internal Application Load
      Balancer.
    networkUrl: The fully qualified URL of the network that the load balancer
      is attached to. This should be formatted like https://www.googleapis.com
      /compute/v1/projects/{project}/global/networks/{network} .
    port: The configured port of the load balancer.
    project: The project ID in which the load balancer is located.
    region: The region in which the load balancer is located.
  """

    class IpProtocolValueValuesEnum(_messages.Enum):
        """The protocol of the load balancer to health check.

    Values:
      undefined: <no description>
      tcp: <no description>
      udp: <no description>
    """
        undefined = 0
        tcp = 1
        udp = 2

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
    ipAddress = _messages.StringField(1)
    ipProtocol = _messages.EnumField('IpProtocolValueValuesEnum', 2)
    kind = _messages.StringField(3, default='dns#rRSetRoutingPolicyLoadBalancerTarget')
    loadBalancerType = _messages.EnumField('LoadBalancerTypeValueValuesEnum', 4)
    networkUrl = _messages.StringField(5)
    port = _messages.StringField(6)
    project = _messages.StringField(7)
    region = _messages.StringField(8)