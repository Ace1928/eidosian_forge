from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheckFirewallStateValueValuesEnum(_messages.Enum):
    """State of the health check firewall configuration.

    Values:
      HEALTH_CHECK_FIREWALL_STATE_UNSPECIFIED: State is unspecified. Default
        state if not populated.
      CONFIGURED: There are configured firewall rules to allow health check
        probes to the backend.
      MISCONFIGURED: There are firewall rules configured to allow partial
        health check ranges or block all health check ranges. If a health
        check probe is sent from denied IP ranges, the health check to the
        backend will fail. Then, the backend will be marked unhealthy and will
        not receive traffic sent to the load balancer.
    """
    HEALTH_CHECK_FIREWALL_STATE_UNSPECIFIED = 0
    CONFIGURED = 1
    MISCONFIGURED = 2