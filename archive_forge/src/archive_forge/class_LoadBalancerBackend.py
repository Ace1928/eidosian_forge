from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadBalancerBackend(_messages.Message):
    """For display only. Metadata associated with a specific load balancer
  backend.

  Enums:
    HealthCheckFirewallStateValueValuesEnum: State of the health check
      firewall configuration.

  Fields:
    displayName: Name of a Compute Engine instance or network endpoint.
    healthCheckAllowingFirewallRules: A list of firewall rule URIs allowing
      probes from health check IP ranges.
    healthCheckBlockingFirewallRules: A list of firewall rule URIs blocking
      probes from health check IP ranges.
    healthCheckFirewallState: State of the health check firewall
      configuration.
    uri: URI of a Compute Engine instance or network endpoint.
  """

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
    displayName = _messages.StringField(1)
    healthCheckAllowingFirewallRules = _messages.StringField(2, repeated=True)
    healthCheckBlockingFirewallRules = _messages.StringField(3, repeated=True)
    healthCheckFirewallState = _messages.EnumField('HealthCheckFirewallStateValueValuesEnum', 4)
    uri = _messages.StringField(5)