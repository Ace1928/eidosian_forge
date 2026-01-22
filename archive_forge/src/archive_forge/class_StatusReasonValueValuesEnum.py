from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatusReasonValueValuesEnum(_messages.Enum):
    """Indicates why particular status was returned.

    Values:
      IPV4_PEER_ON_IPV6_ONLY_CONNECTION: BGP peer disabled because it requires
        IPv4 but the underlying connection is IPv6-only.
      IPV6_PEER_ON_IPV4_ONLY_CONNECTION: BGP peer disabled because it requires
        IPv6 but the underlying connection is IPv4-only.
      MD5_AUTH_INTERNAL_PROBLEM: Indicates internal problems with
        configuration of MD5 authentication. This particular reason can only
        be returned when md5AuthEnabled is true and status is DOWN.
      STATUS_REASON_UNSPECIFIED: <no description>
    """
    IPV4_PEER_ON_IPV6_ONLY_CONNECTION = 0
    IPV6_PEER_ON_IPV4_ONLY_CONNECTION = 1
    MD5_AUTH_INTERNAL_PROBLEM = 2
    STATUS_REASON_UNSPECIFIED = 3