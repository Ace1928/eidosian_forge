from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadBalancingAlgorithmValueValuesEnum(_messages.Enum):
    """Optional. The type of load balancing algorithm to be used. The default
    behavior is WATERFALL_BY_REGION.

    Values:
      LOAD_BALANCING_ALGORITHM_UNSPECIFIED: The type of the loadbalancing
        algorithm is unspecified.
      SPRAY_TO_WORLD: Balance traffic across all backends across the world
        proportionally based on capacity.
      SPRAY_TO_REGION: Direct traffic to the nearest region with endpoints and
        capacity before spilling over to other regions and spread the traffic
        from each client to all the MIGs/NEGs in a region.
      WATERFALL_BY_REGION: Direct traffic to the nearest region with endpoints
        and capacity before spilling over to other regions. All MIGs/NEGs
        within a region are evenly loaded but each client might not spread the
        traffic to all the MIGs/NEGs in the region.
      WATERFALL_BY_ZONE: Attempt to keep traffic in a single zone closest to
        the client, before spilling over to other zones.
    """
    LOAD_BALANCING_ALGORITHM_UNSPECIFIED = 0
    SPRAY_TO_WORLD = 1
    SPRAY_TO_REGION = 2
    WATERFALL_BY_REGION = 3
    WATERFALL_BY_ZONE = 4