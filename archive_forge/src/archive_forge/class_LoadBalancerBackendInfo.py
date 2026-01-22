from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadBalancerBackendInfo(_messages.Message):
    """For display only. Metadata associated with the load balancer backend.

  Enums:
    HealthCheckFirewallsConfigStateValueValuesEnum: Output only. Health check
      firewalls configuration state for the backend. This is a result of the
      static firewall analysis (verifying that health check traffic from
      required IP ranges to the backend is allowed or not). The backend might
      still be unhealthy even if these firewalls are configured. Please refer
      to the documentation for more information:
      https://cloud.google.com/load-balancing/docs/firewall-rules

  Fields:
    backendBucketUri: URI of the backend bucket this backend targets (if
      applicable).
    backendServiceUri: URI of the backend service this backend belongs to (if
      applicable).
    healthCheckFirewallsConfigState: Output only. Health check firewalls
      configuration state for the backend. This is a result of the static
      firewall analysis (verifying that health check traffic from required IP
      ranges to the backend is allowed or not). The backend might still be
      unhealthy even if these firewalls are configured. Please refer to the
      documentation for more information: https://cloud.google.com/load-
      balancing/docs/firewall-rules
    healthCheckUri: URI of the health check attached to this backend (if
      applicable).
    instanceGroupUri: URI of the instance group this backend belongs to (if
      applicable).
    instanceUri: URI of the backend instance (if applicable). Populated for
      instance group backends, and zonal NEG backends.
    name: Display name of the backend. For example, it might be an instance
      name for the instance group backends, or an IP address and port for
      zonal network endpoint group backends.
    networkEndpointGroupUri: URI of the network endpoint group this backend
      belongs to (if applicable).
    pscGoogleApiTarget: PSC Google API target this PSC NEG backend targets (if
      applicable).
    pscServiceAttachmentUri: URI of the PSC service attachment this PSC NEG
      backend targets (if applicable).
  """

    class HealthCheckFirewallsConfigStateValueValuesEnum(_messages.Enum):
        """Output only. Health check firewalls configuration state for the
    backend. This is a result of the static firewall analysis (verifying that
    health check traffic from required IP ranges to the backend is allowed or
    not). The backend might still be unhealthy even if these firewalls are
    configured. Please refer to the documentation for more information:
    https://cloud.google.com/load-balancing/docs/firewall-rules

    Values:
      HEALTH_CHECK_FIREWALLS_CONFIG_STATE_UNSPECIFIED: Configuration state
        unspecified. It usually means that the backend has no health check
        attached, or there was an unexpected configuration error preventing
        Connectivity tests from verifying health check configuration.
      FIREWALLS_CONFIGURED: Firewall rules (policies) allowing health check
        traffic from all required IP ranges to the backend are configured.
      FIREWALLS_PARTIALLY_CONFIGURED: Firewall rules (policies) allow health
        check traffic only from a part of required IP ranges.
      FIREWALLS_NOT_CONFIGURED: Firewall rules (policies) deny health check
        traffic from all required IP ranges to the backend.
      FIREWALLS_UNSUPPORTED: The network contains firewall rules of
        unsupported types, so Connectivity tests were not able to verify
        health check configuration status. Please refer to the documentation
        for the list of unsupported configurations:
        https://cloud.google.com/network-intelligence-
        center/docs/connectivity-tests/concepts/overview#unsupported-configs
    """
        HEALTH_CHECK_FIREWALLS_CONFIG_STATE_UNSPECIFIED = 0
        FIREWALLS_CONFIGURED = 1
        FIREWALLS_PARTIALLY_CONFIGURED = 2
        FIREWALLS_NOT_CONFIGURED = 3
        FIREWALLS_UNSUPPORTED = 4
    backendBucketUri = _messages.StringField(1)
    backendServiceUri = _messages.StringField(2)
    healthCheckFirewallsConfigState = _messages.EnumField('HealthCheckFirewallsConfigStateValueValuesEnum', 3)
    healthCheckUri = _messages.StringField(4)
    instanceGroupUri = _messages.StringField(5)
    instanceUri = _messages.StringField(6)
    name = _messages.StringField(7)
    networkEndpointGroupUri = _messages.StringField(8)
    pscGoogleApiTarget = _messages.StringField(9)
    pscServiceAttachmentUri = _messages.StringField(10)