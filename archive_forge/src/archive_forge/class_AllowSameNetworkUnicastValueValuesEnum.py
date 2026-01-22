from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowSameNetworkUnicastValueValuesEnum(_messages.Enum):
    """Specifies whether unicast within the same network is allowed.

    Values:
      SAME_NETWORK_UNICAST_ALLOWED: <no description>
      SAME_NETWORK_UNICAST_BLOCKED: <no description>
      SAME_NETWORK_UNICAST_UNSPECIFIED: <no description>
    """
    SAME_NETWORK_UNICAST_ALLOWED = 0
    SAME_NETWORK_UNICAST_BLOCKED = 1
    SAME_NETWORK_UNICAST_UNSPECIFIED = 2