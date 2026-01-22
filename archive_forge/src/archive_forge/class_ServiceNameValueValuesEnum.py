from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceNameValueValuesEnum(_messages.Enum):
    """Required. internal service name.

    Values:
      INTERNAL_OS_SERVICE_ENUM_UNSPECIFIED: Service name unknown.
      DOCKER_SERVICE_STATE: Represents the internal os docker client.
      CONTROL_PLANE_API_DNS_STATE: Represents reoslving DNS for the control
        plane api endpoint.
      PROXY_REGISTRATION_DNS_STATE: Represents reoslving DNS for the proxy
        registration endpoint.
      JUPYTER_STATE: Represents the jupyter endpoint.
      JUPYTER_API_STATE: Represents the jupyter/api endpoint.
      EUC_METADATA_API_STATE: Represents the EUC metadata server API endpoint.
      EUC_AGENT_API_STATE: Represents the EUC agent server API endpoint.
      IDLE_SHUTDOWN_AGENT_STATE: Represents the idle shutdown agent sidecar
        container.
      PROXY_AGENT_STATE: Represents the proxy agent sidecar container.
    """
    INTERNAL_OS_SERVICE_ENUM_UNSPECIFIED = 0
    DOCKER_SERVICE_STATE = 1
    CONTROL_PLANE_API_DNS_STATE = 2
    PROXY_REGISTRATION_DNS_STATE = 3
    JUPYTER_STATE = 4
    JUPYTER_API_STATE = 5
    EUC_METADATA_API_STATE = 6
    EUC_AGENT_API_STATE = 7
    IDLE_SHUTDOWN_AGENT_STATE = 8
    PROXY_AGENT_STATE = 9