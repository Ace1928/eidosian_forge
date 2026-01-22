from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutingTypeValueValuesEnum(_messages.Enum):
    """Type of the routing policy.

    Values:
      ROUTING_TYPE_UNSPECIFIED: Unspecified type. Default value.
      ROUTE_BASED: Route based VPN.
      POLICY_BASED: Policy based routing.
      DYNAMIC: Dynamic (BGP) routing.
    """
    ROUTING_TYPE_UNSPECIFIED = 0
    ROUTE_BASED = 1
    POLICY_BASED = 2
    DYNAMIC = 3