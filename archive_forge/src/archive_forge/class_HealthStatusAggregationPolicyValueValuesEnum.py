from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthStatusAggregationPolicyValueValuesEnum(_messages.Enum):
    """Optional. Policy for how the results from multiple health checks for
    the same endpoint are aggregated. Defaults to NO_AGGREGATION if
    unspecified. - NO_AGGREGATION. An EndpointHealth message is returned for
    each pair in the health check service. - AND. If any health check of an
    endpoint reports UNHEALTHY, then UNHEALTHY is the HealthState of the
    endpoint. If all health checks report HEALTHY, the HealthState of the
    endpoint is HEALTHY. . This is only allowed with regional
    HealthCheckService.

    Values:
      AND: If any backend's health check reports UNHEALTHY, then UNHEALTHY is
        the HealthState of the entire health check service. If all backend's
        are healthy, the HealthState of the health check service is HEALTHY.
      NO_AGGREGATION: An EndpointHealth message is returned for each backend
        in the health check service.
    """
    AND = 0
    NO_AGGREGATION = 1