from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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