from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetHttpsProxy(_messages.Message):
    """Represents a Target HTTPS Proxy resource. Google Compute Engine has two
  Target HTTPS Proxy resources: *
  [Global](/compute/docs/reference/rest/beta/targetHttpsProxies) *
  [Regional](/compute/docs/reference/rest/beta/regionTargetHttpsProxies) A
  target HTTPS proxy is a component of GCP HTTPS load balancers. *
  targetHttpProxies are used by global external Application Load Balancers,
  classic Application Load Balancers, cross-region internal Application Load
  Balancers, and Traffic Director. * regionTargetHttpProxies are used by
  regional internal Application Load Balancers and regional external
  Application Load Balancers. Forwarding rules reference a target HTTPS proxy,
  and the target proxy then references a URL map. For more information, read
  Using Target Proxies and Forwarding rule concepts.

  Enums:
    QuicOverrideValueValuesEnum: Specifies the QUIC override policy for this
      TargetHttpsProxy resource. This setting determines whether the load
      balancer attempts to negotiate QUIC with clients. You can specify NONE,
      ENABLE, or DISABLE. - When quic-override is set to NONE, Google manages
      whether QUIC is used. - When quic-override is set to ENABLE, the load
      balancer uses QUIC when possible. - When quic-override is set to
      DISABLE, the load balancer doesn't use QUIC. - If the quic-override flag
      is not specified, NONE is implied.
    TlsEarlyDataValueValuesEnum:  Specifies whether TLS 1.3 0-RTT Data ("Early
      Data") should be accepted for this service. Early Data allows a TLS
      resumption handshake to include the initial application payload (a HTTP
      request) alongside the handshake, reducing the effective round trips to
      "zero". This applies to TLS 1.3 connections over TCP (HTTP/2) as well as
      over UDP (QUIC/h3). This can improve application performance, especially
      on networks where interruptions may be common, such as on mobile.
      Requests with Early Data will have the "Early-Data" HTTP header set on
      the request, with a value of "1", to allow the backend to determine
      whether Early Data was included. Note: TLS Early Data may allow requests
      to be replayed, as the data is sent to the backend before the handshake
      has fully completed. Applications that allow idempotent HTTP methods to
      make non-idempotent changes, such as a GET request updating a database,
      should not accept Early Data on those requests, and reject requests with
      the "Early-Data: 1" HTTP header by returning a HTTP 425 (Too Early)
      status code, in order to remain RFC compliant. The default value is
      DISABLED.

  Fields:
    authentication: [Deprecated] Use serverTlsPolicy instead.
    authorization: [Deprecated] Use authorizationPolicy instead.
    authorizationPolicy: Optional. A URL referring to a
      networksecurity.AuthorizationPolicy resource that describes how the
      proxy should authorize inbound traffic. If left blank, access will not
      be restricted by an authorization policy. Refer to the
      AuthorizationPolicy resource for additional details. authorizationPolicy
      only applies to a global TargetHttpsProxy attached to
      globalForwardingRules with the loadBalancingScheme set to
      INTERNAL_SELF_MANAGED. Note: This field currently has no impact.
    certificateMap: URL of a certificate map that identifies a certificate map
      associated with the given target proxy. This field can only be set for
      global target proxies. If set, sslCertificates will be ignored. Accepted
      format is //certificatemanager.googleapis.com/projects/{project
      }/locations/{location}/certificateMaps/{resourceName}.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a TargetHttpsProxy. An up-to-date
      fingerprint must be provided in order to patch the TargetHttpsProxy;
      otherwise, the request will fail with error 412 conditionNotMet. To see
      the latest fingerprint, make a get() request to retrieve the
      TargetHttpsProxy.
    httpFilters: URLs to networkservices.HttpFilter resources enabled for xDS
      clients using this configuration. For example,
      https://networkservices.googleapis.com/beta/projects/project/locations/
      locationhttpFilters/httpFilter Only filters that handle outbound
      connection and stream events may be specified. These filters work in
      conjunction with a default set of HTTP filters that may already be
      configured by Traffic Director. Traffic Director will determine the
      final location of these filters within xDS configuration based on the
      name of the HTTP filter. If Traffic Director positions multiple filters
      at the same location, those filters will be in the same order as
      specified in this list. httpFilters only applies for loadbalancers with
      loadBalancingScheme set to INTERNAL_SELF_MANAGED. See ForwardingRule for
      more details.
    httpKeepAliveTimeoutSec: Specifies how long to keep a connection open,
      after completing a response, while there is no matching traffic (in
      seconds). If an HTTP keep-alive is not specified, a default value (610
      seconds) will be used. For global external Application Load Balancers,
      the minimum allowed value is 5 seconds and the maximum allowed value is
      1200 seconds. For classic Application Load Balancers, this option is not
      supported.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of resource. Always compute#targetHttpsProxy for
      target HTTPS proxies.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    proxyBind: This field only applies when the forwarding rule that
      references this target proxy has a loadBalancingScheme set to
      INTERNAL_SELF_MANAGED. When this field is set to true, Envoy proxies set
      up inbound traffic interception and bind to the IP address and port
      specified in the forwarding rule. This is generally useful when using
      Traffic Director to configure Envoy as a gateway or middle proxy (in
      other words, not a sidecar proxy). The Envoy proxy listens for inbound
      requests and handles requests when it receives them. The default is
      false.
    quicOverride: Specifies the QUIC override policy for this TargetHttpsProxy
      resource. This setting determines whether the load balancer attempts to
      negotiate QUIC with clients. You can specify NONE, ENABLE, or DISABLE. -
      When quic-override is set to NONE, Google manages whether QUIC is used.
      - When quic-override is set to ENABLE, the load balancer uses QUIC when
      possible. - When quic-override is set to DISABLE, the load balancer
      doesn't use QUIC. - If the quic-override flag is not specified, NONE is
      implied.
    region: [Output Only] URL of the region where the regional
      TargetHttpsProxy resides. This field is not applicable to global
      TargetHttpsProxies.
    selfLink: [Output Only] Server-defined URL for the resource.
    serverTlsPolicy: Optional. A URL referring to a
      networksecurity.ServerTlsPolicy resource that describes how the proxy
      should authenticate inbound traffic. serverTlsPolicy only applies to a
      global TargetHttpsProxy attached to globalForwardingRules with the
      loadBalancingScheme set to INTERNAL_SELF_MANAGED or EXTERNAL or
      EXTERNAL_MANAGED. For details which ServerTlsPolicy resources are
      accepted with INTERNAL_SELF_MANAGED and which with EXTERNAL,
      EXTERNAL_MANAGED loadBalancingScheme consult ServerTlsPolicy
      documentation. If left blank, communications are not encrypted.
    sslCertificates: URLs to SslCertificate resources that are used to
      authenticate connections between users and the load balancer. At least
      one SSL certificate must be specified. Currently, you may specify up to
      15 SSL certificates. sslCertificates do not apply when the load
      balancing scheme is set to INTERNAL_SELF_MANAGED.
    sslPolicy: URL of SslPolicy resource that will be associated with the
      TargetHttpsProxy resource. If not set, the TargetHttpsProxy resource has
      no SSL policy configured.
    tlsEarlyData:  Specifies whether TLS 1.3 0-RTT Data ("Early Data") should
      be accepted for this service. Early Data allows a TLS resumption
      handshake to include the initial application payload (a HTTP request)
      alongside the handshake, reducing the effective round trips to "zero".
      This applies to TLS 1.3 connections over TCP (HTTP/2) as well as over
      UDP (QUIC/h3). This can improve application performance, especially on
      networks where interruptions may be common, such as on mobile. Requests
      with Early Data will have the "Early-Data" HTTP header set on the
      request, with a value of "1", to allow the backend to determine whether
      Early Data was included. Note: TLS Early Data may allow requests to be
      replayed, as the data is sent to the backend before the handshake has
      fully completed. Applications that allow idempotent HTTP methods to make
      non-idempotent changes, such as a GET request updating a database,
      should not accept Early Data on those requests, and reject requests with
      the "Early-Data: 1" HTTP header by returning a HTTP 425 (Too Early)
      status code, in order to remain RFC compliant. The default value is
      DISABLED.
    urlMap: A fully-qualified or valid partial URL to the UrlMap resource that
      defines the mapping from URL to the BackendService. For example, the
      following are all valid URLs for specifying a URL map: -
      https://www.googleapis.compute/v1/projects/project/global/urlMaps/ url-
      map - projects/project/global/urlMaps/url-map - global/urlMaps/url-map
  """

    class QuicOverrideValueValuesEnum(_messages.Enum):
        """Specifies the QUIC override policy for this TargetHttpsProxy resource.
    This setting determines whether the load balancer attempts to negotiate
    QUIC with clients. You can specify NONE, ENABLE, or DISABLE. - When quic-
    override is set to NONE, Google manages whether QUIC is used. - When quic-
    override is set to ENABLE, the load balancer uses QUIC when possible. -
    When quic-override is set to DISABLE, the load balancer doesn't use QUIC.
    - If the quic-override flag is not specified, NONE is implied.

    Values:
      DISABLE: The load balancer will not attempt to negotiate QUIC with
        clients.
      ENABLE: The load balancer will attempt to negotiate QUIC with clients.
      NONE: No overrides to the default QUIC policy. This option is implicit
        if no QUIC override has been specified in the request.
    """
        DISABLE = 0
        ENABLE = 1
        NONE = 2

    class TlsEarlyDataValueValuesEnum(_messages.Enum):
        """ Specifies whether TLS 1.3 0-RTT Data ("Early Data") should be
    accepted for this service. Early Data allows a TLS resumption handshake to
    include the initial application payload (a HTTP request) alongside the
    handshake, reducing the effective round trips to "zero". This applies to
    TLS 1.3 connections over TCP (HTTP/2) as well as over UDP (QUIC/h3). This
    can improve application performance, especially on networks where
    interruptions may be common, such as on mobile. Requests with Early Data
    will have the "Early-Data" HTTP header set on the request, with a value of
    "1", to allow the backend to determine whether Early Data was included.
    Note: TLS Early Data may allow requests to be replayed, as the data is
    sent to the backend before the handshake has fully completed. Applications
    that allow idempotent HTTP methods to make non-idempotent changes, such as
    a GET request updating a database, should not accept Early Data on those
    requests, and reject requests with the "Early-Data: 1" HTTP header by
    returning a HTTP 425 (Too Early) status code, in order to remain RFC
    compliant. The default value is DISABLED.

    Values:
      DISABLED: TLS 1.3 Early Data is not advertised, and any (invalid)
        attempts to send Early Data will be rejected by closing the
        connection.
      PERMISSIVE: This enables TLS 1.3 0-RTT, and only allows Early Data to be
        included on requests with safe HTTP methods (GET, HEAD, OPTIONS,
        TRACE). This mode does not enforce any other limitations for requests
        with Early Data. The application owner should validate that Early Data
        is acceptable for a given request path.
      STRICT: This enables TLS 1.3 0-RTT, and only allows Early Data to be
        included on requests with safe HTTP methods (GET, HEAD, OPTIONS,
        TRACE) without query parameters. Requests that send Early Data with
        non-idempotent HTTP methods or with query parameters will be rejected
        with a HTTP 425.
    """
        DISABLED = 0
        PERMISSIVE = 1
        STRICT = 2
    authentication = _messages.StringField(1)
    authorization = _messages.StringField(2)
    authorizationPolicy = _messages.StringField(3)
    certificateMap = _messages.StringField(4)
    creationTimestamp = _messages.StringField(5)
    description = _messages.StringField(6)
    fingerprint = _messages.BytesField(7)
    httpFilters = _messages.StringField(8, repeated=True)
    httpKeepAliveTimeoutSec = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    id = _messages.IntegerField(10, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(11, default='compute#targetHttpsProxy')
    name = _messages.StringField(12)
    proxyBind = _messages.BooleanField(13)
    quicOverride = _messages.EnumField('QuicOverrideValueValuesEnum', 14)
    region = _messages.StringField(15)
    selfLink = _messages.StringField(16)
    serverTlsPolicy = _messages.StringField(17)
    sslCertificates = _messages.StringField(18, repeated=True)
    sslPolicy = _messages.StringField(19)
    tlsEarlyData = _messages.EnumField('TlsEarlyDataValueValuesEnum', 20)
    urlMap = _messages.StringField(21)