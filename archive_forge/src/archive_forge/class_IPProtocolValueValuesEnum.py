from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IPProtocolValueValuesEnum(_messages.Enum):
    """The IP protocol to which this rule applies. For protocol forwarding,
    valid options are TCP, UDP, ESP, AH, SCTP, ICMP and L3_DEFAULT. The valid
    IP protocols are different for different load balancing products as
    described in [Load balancing features](https://cloud.google.com/load-
    balancing/docs/features#protocols_from_the_load_balancer_to_the_backends).

    Values:
      AH: <no description>
      ESP: <no description>
      ICMP: <no description>
      L3_DEFAULT: <no description>
      SCTP: <no description>
      TCP: <no description>
      UDP: <no description>
    """
    AH = 0
    ESP = 1
    ICMP = 2
    L3_DEFAULT = 3
    SCTP = 4
    TCP = 5
    UDP = 6