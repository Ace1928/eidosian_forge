from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceNetworkPassThroughLbTrafficPolicyZonalAffinity(_messages.Message):
    """A BackendServiceNetworkPassThroughLbTrafficPolicyZonalAffinity object.

  Enums:
    SpilloverValueValuesEnum: This field indicates whether zonal affinity is
      enabled or not. The possible values are: - ZONAL_AFFINITY_DISABLED:
      Default Value. Zonal Affinity is disabled. The load balancer distributes
      new connections to all healthy backend endpoints across all zones. -
      ZONAL_AFFINITY_STAY_WITHIN_ZONE: Zonal Affinity is enabled. The load
      balancer distributes new connections to all healthy backend endpoints in
      the local zone only. If there are no healthy backend endpoints in the
      local zone, the load balancer distributes new connections to all backend
      endpoints in the local zone. - ZONAL_AFFINITY_SPILL_CROSS_ZONE: Zonal
      Affinity is enabled. The load balancer distributes new connections to
      all healthy backend endpoints in the local zone only. If there aren't
      enough healthy backend endpoints in the local zone, the load balancer
      distributes new connections to all healthy backend endpoints across all
      zones.

  Fields:
    spillover: This field indicates whether zonal affinity is enabled or not.
      The possible values are: - ZONAL_AFFINITY_DISABLED: Default Value. Zonal
      Affinity is disabled. The load balancer distributes new connections to
      all healthy backend endpoints across all zones. -
      ZONAL_AFFINITY_STAY_WITHIN_ZONE: Zonal Affinity is enabled. The load
      balancer distributes new connections to all healthy backend endpoints in
      the local zone only. If there are no healthy backend endpoints in the
      local zone, the load balancer distributes new connections to all backend
      endpoints in the local zone. - ZONAL_AFFINITY_SPILL_CROSS_ZONE: Zonal
      Affinity is enabled. The load balancer distributes new connections to
      all healthy backend endpoints in the local zone only. If there aren't
      enough healthy backend endpoints in the local zone, the load balancer
      distributes new connections to all healthy backend endpoints across all
      zones.
    spilloverRatio: The value of the field must be in [0, 1]. When the ratio
      of the count of healthy backend endpoints in a zone to the count of
      backend endpoints in that same zone is equal to or above this threshold,
      the load balancer distributes new connections to all healthy endpoints
      in the local zone only. When the ratio of the count of healthy backend
      endpoints in a zone to the count of backend endpoints in that same zone
      is below this threshold, the load balancer distributes all new
      connections to all healthy endpoints across all zones.
  """

    class SpilloverValueValuesEnum(_messages.Enum):
        """This field indicates whether zonal affinity is enabled or not. The
    possible values are: - ZONAL_AFFINITY_DISABLED: Default Value. Zonal
    Affinity is disabled. The load balancer distributes new connections to all
    healthy backend endpoints across all zones. -
    ZONAL_AFFINITY_STAY_WITHIN_ZONE: Zonal Affinity is enabled. The load
    balancer distributes new connections to all healthy backend endpoints in
    the local zone only. If there are no healthy backend endpoints in the
    local zone, the load balancer distributes new connections to all backend
    endpoints in the local zone. - ZONAL_AFFINITY_SPILL_CROSS_ZONE: Zonal
    Affinity is enabled. The load balancer distributes new connections to all
    healthy backend endpoints in the local zone only. If there aren't enough
    healthy backend endpoints in the local zone, the load balancer distributes
    new connections to all healthy backend endpoints across all zones.

    Values:
      ZONAL_AFFINITY_DISABLED: <no description>
      ZONAL_AFFINITY_SPILL_CROSS_ZONE: <no description>
      ZONAL_AFFINITY_STAY_WITHIN_ZONE: <no description>
    """
        ZONAL_AFFINITY_DISABLED = 0
        ZONAL_AFFINITY_SPILL_CROSS_ZONE = 1
        ZONAL_AFFINITY_STAY_WITHIN_ZONE = 2
    spillover = _messages.EnumField('SpilloverValueValuesEnum', 1)
    spilloverRatio = _messages.FloatField(2, variant=_messages.Variant.FLOAT)