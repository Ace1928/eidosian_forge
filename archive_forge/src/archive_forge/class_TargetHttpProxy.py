from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetHttpProxy(_messages.Message):
    """Represents a Target HTTP Proxy resource. Google Compute Engine has two
  Target HTTP Proxy resources: *
  [Global](/compute/docs/reference/rest/beta/targetHttpProxies) *
  [Regional](/compute/docs/reference/rest/beta/regionTargetHttpProxies) A
  target HTTP proxy is a component of Google Cloud HTTP load balancers. *
  targetHttpProxies are used by global external Application Load Balancers,
  classic Application Load Balancers, cross-region internal Application Load
  Balancers, and Traffic Director. * regionTargetHttpProxies are used by
  regional internal Application Load Balancers and regional external
  Application Load Balancers. Forwarding rules reference a target HTTP proxy,
  and the target proxy then references a URL map. For more information, read
  Using Target Proxies and Forwarding rule concepts.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a TargetHttpProxy. An up-to-date
      fingerprint must be provided in order to patch/update the
      TargetHttpProxy; otherwise, the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve the TargetHttpProxy.
    httpFilters: URLs to networkservices.HttpFilter resources enabled for xDS
      clients using this configuration. For example, https://networkservices.g
      oogleapis.com/v1alpha1/projects/project/locations/
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
    kind: [Output Only] Type of resource. Always compute#targetHttpProxy for
      target HTTP proxies.
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
    region: [Output Only] URL of the region where the regional Target HTTP
      Proxy resides. This field is not applicable to global Target HTTP
      Proxies.
    selfLink: [Output Only] Server-defined URL for the resource.
    urlMap: URL to the UrlMap resource that defines the mapping from URL to
      the BackendService.
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    fingerprint = _messages.BytesField(3)
    httpFilters = _messages.StringField(4, repeated=True)
    httpKeepAliveTimeoutSec = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(7, default='compute#targetHttpProxy')
    name = _messages.StringField(8)
    proxyBind = _messages.BooleanField(9)
    region = _messages.StringField(10)
    selfLink = _messages.StringField(11)
    urlMap = _messages.StringField(12)