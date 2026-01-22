from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadBalancerInfo(_messages.Message):
    """For display only. Metadata associated with a load balancer.

  Enums:
    BackendTypeValueValuesEnum: Type of load balancer's backend configuration.
    LoadBalancerTypeValueValuesEnum: Type of the load balancer.

  Fields:
    backendType: Type of load balancer's backend configuration.
    backendUri: Backend configuration URI.
    backends: Information for the loadbalancer backends.
    healthCheckUri: URI of the health check for the load balancer. Deprecated
      and no longer populated as different load balancer backends might have
      different health checks.
    loadBalancerType: Type of the load balancer.
  """

    class BackendTypeValueValuesEnum(_messages.Enum):
        """Type of load balancer's backend configuration.

    Values:
      BACKEND_TYPE_UNSPECIFIED: Type is unspecified.
      BACKEND_SERVICE: Backend Service as the load balancer's backend.
      TARGET_POOL: Target Pool as the load balancer's backend.
      TARGET_INSTANCE: Target Instance as the load balancer's backend.
    """
        BACKEND_TYPE_UNSPECIFIED = 0
        BACKEND_SERVICE = 1
        TARGET_POOL = 2
        TARGET_INSTANCE = 3

    class LoadBalancerTypeValueValuesEnum(_messages.Enum):
        """Type of the load balancer.

    Values:
      LOAD_BALANCER_TYPE_UNSPECIFIED: Type is unspecified.
      INTERNAL_TCP_UDP: Internal TCP/UDP load balancer.
      NETWORK_TCP_UDP: Network TCP/UDP load balancer.
      HTTP_PROXY: HTTP(S) proxy load balancer.
      TCP_PROXY: TCP proxy load balancer.
      SSL_PROXY: SSL proxy load balancer.
    """
        LOAD_BALANCER_TYPE_UNSPECIFIED = 0
        INTERNAL_TCP_UDP = 1
        NETWORK_TCP_UDP = 2
        HTTP_PROXY = 3
        TCP_PROXY = 4
        SSL_PROXY = 5
    backendType = _messages.EnumField('BackendTypeValueValuesEnum', 1)
    backendUri = _messages.StringField(2)
    backends = _messages.MessageField('LoadBalancerBackend', 3, repeated=True)
    healthCheckUri = _messages.StringField(4)
    loadBalancerType = _messages.EnumField('LoadBalancerTypeValueValuesEnum', 5)