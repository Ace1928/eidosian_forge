from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HTTPSHealthCheck(_messages.Message):
    """A HTTPSHealthCheck object.

  Enums:
    PortSpecificationValueValuesEnum: Specifies how a port is selected for
      health checking. Can be one of the following values: USE_FIXED_PORT:
      Specifies a port number explicitly using the port field in the health
      check. Supported by backend services for passthrough load balancers and
      backend services for proxy load balancers. Not supported by target
      pools. The health check supports all backends supported by the backend
      service provided the backend can be health checked. For example,
      GCE_VM_IP network endpoint groups, GCE_VM_IP_PORT network endpoint
      groups, and instance group backends. USE_NAMED_PORT: Not supported.
      USE_SERVING_PORT: Provides an indirect method of specifying the health
      check port by referring to the backend service. Only supported by
      backend services for proxy load balancers. Not supported by target
      pools. Not supported by backend services for passthrough load balancers.
      Supports all backends that can be health checked; for example,
      GCE_VM_IP_PORT network endpoint groups and instance group backends. For
      GCE_VM_IP_PORT network endpoint group backends, the health check uses
      the port number specified for each endpoint in the network endpoint
      group. For instance group backends, the health check uses the port
      number determined by looking up the backend service's named port in the
      instance group's list of named ports.
    ProxyHeaderValueValuesEnum: Specifies the type of proxy header to append
      before sending data to the backend, either NONE or PROXY_V1. The default
      is NONE.

  Fields:
    host: The value of the host header in the HTTPS health check request. If
      left empty (default value), the host header is set to the destination IP
      address to which health check packets are sent. The destination IP
      address depends on the type of load balancer. For details, see:
      https://cloud.google.com/load-balancing/docs/health-check-concepts#hc-
      packet-dest
    port: The TCP port number to which the health check prober sends packets.
      The default value is 443. Valid values are 1 through 65535.
    portName: Not supported.
    portSpecification: Specifies how a port is selected for health checking.
      Can be one of the following values: USE_FIXED_PORT: Specifies a port
      number explicitly using the port field in the health check. Supported by
      backend services for passthrough load balancers and backend services for
      proxy load balancers. Not supported by target pools. The health check
      supports all backends supported by the backend service provided the
      backend can be health checked. For example, GCE_VM_IP network endpoint
      groups, GCE_VM_IP_PORT network endpoint groups, and instance group
      backends. USE_NAMED_PORT: Not supported. USE_SERVING_PORT: Provides an
      indirect method of specifying the health check port by referring to the
      backend service. Only supported by backend services for proxy load
      balancers. Not supported by target pools. Not supported by backend
      services for passthrough load balancers. Supports all backends that can
      be health checked; for example, GCE_VM_IP_PORT network endpoint groups
      and instance group backends. For GCE_VM_IP_PORT network endpoint group
      backends, the health check uses the port number specified for each
      endpoint in the network endpoint group. For instance group backends, the
      health check uses the port number determined by looking up the backend
      service's named port in the instance group's list of named ports.
    proxyHeader: Specifies the type of proxy header to append before sending
      data to the backend, either NONE or PROXY_V1. The default is NONE.
    requestPath: The request path of the HTTPS health check request. The
      default value is /.
    response: Creates a content-based HTTPS health check. In addition to the
      required HTTP 200 (OK) status code, you can configure the health check
      to pass only when the backend sends this specific ASCII response string
      within the first 1024 bytes of the HTTP response body. For details, see:
      https://cloud.google.com/load-balancing/docs/health-check-
      concepts#criteria-protocol-http
  """

    class PortSpecificationValueValuesEnum(_messages.Enum):
        """Specifies how a port is selected for health checking. Can be one of
    the following values: USE_FIXED_PORT: Specifies a port number explicitly
    using the port field in the health check. Supported by backend services
    for passthrough load balancers and backend services for proxy load
    balancers. Not supported by target pools. The health check supports all
    backends supported by the backend service provided the backend can be
    health checked. For example, GCE_VM_IP network endpoint groups,
    GCE_VM_IP_PORT network endpoint groups, and instance group backends.
    USE_NAMED_PORT: Not supported. USE_SERVING_PORT: Provides an indirect
    method of specifying the health check port by referring to the backend
    service. Only supported by backend services for proxy load balancers. Not
    supported by target pools. Not supported by backend services for
    passthrough load balancers. Supports all backends that can be health
    checked; for example, GCE_VM_IP_PORT network endpoint groups and instance
    group backends. For GCE_VM_IP_PORT network endpoint group backends, the
    health check uses the port number specified for each endpoint in the
    network endpoint group. For instance group backends, the health check uses
    the port number determined by looking up the backend service's named port
    in the instance group's list of named ports.

    Values:
      USE_FIXED_PORT: The port number in the health check's port is used for
        health checking. Applies to network endpoint group and instance group
        backends.
      USE_NAMED_PORT: Not supported.
      USE_SERVING_PORT: For network endpoint group backends, the health check
        uses the port number specified on each endpoint in the network
        endpoint group. For instance group backends, the health check uses the
        port number specified for the backend service's named port defined in
        the instance group's named ports.
    """
        USE_FIXED_PORT = 0
        USE_NAMED_PORT = 1
        USE_SERVING_PORT = 2

    class ProxyHeaderValueValuesEnum(_messages.Enum):
        """Specifies the type of proxy header to append before sending data to
    the backend, either NONE or PROXY_V1. The default is NONE.

    Values:
      NONE: <no description>
      PROXY_V1: <no description>
    """
        NONE = 0
        PROXY_V1 = 1
    host = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    portName = _messages.StringField(3)
    portSpecification = _messages.EnumField('PortSpecificationValueValuesEnum', 4)
    proxyHeader = _messages.EnumField('ProxyHeaderValueValuesEnum', 5)
    requestPath = _messages.StringField(6)
    response = _messages.StringField(7)