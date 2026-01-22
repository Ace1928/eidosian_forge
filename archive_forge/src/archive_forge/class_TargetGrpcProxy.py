from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetGrpcProxy(_messages.Message):
    """Represents a Target gRPC Proxy resource. A target gRPC proxy is a
  component of load balancers intended for load balancing gRPC traffic. Only
  global forwarding rules with load balancing scheme INTERNAL_SELF_MANAGED can
  reference a target gRPC proxy. The target gRPC Proxy references a URL map
  that specifies how traffic is routed to gRPC backend services.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a TargetGrpcProxy. An up-to-date
      fingerprint must be provided in order to patch/update the
      TargetGrpcProxy; otherwise, the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve the TargetGrpcProxy.
    id: [Output Only] The unique identifier for the resource type. The server
      generates this identifier.
    kind: [Output Only] Type of the resource. Always compute#targetGrpcProxy
      for target grpc proxies.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    selfLink: [Output Only] Server-defined URL for the resource.
    selfLinkWithId: [Output Only] Server-defined URL with id for the resource.
    urlMap: URL to the UrlMap resource that defines the mapping from URL to
      the BackendService. The protocol field in the BackendService must be set
      to GRPC.
    validateForProxyless: If true, indicates that the BackendServices
      referenced by the urlMap may be accessed by gRPC applications without
      using a sidecar proxy. This will enable configuration checks on urlMap
      and its referenced BackendServices to not allow unsupported features. A
      gRPC application must use "xds:///" scheme in the target URI of the
      service it is connecting to. If false, indicates that the
      BackendServices referenced by the urlMap will be accessed by gRPC
      applications via a sidecar proxy. In this case, a gRPC application must
      not use "xds:///" scheme in the target URI of the service it is
      connecting to
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    fingerprint = _messages.BytesField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#targetGrpcProxy')
    name = _messages.StringField(6)
    selfLink = _messages.StringField(7)
    selfLinkWithId = _messages.StringField(8)
    urlMap = _messages.StringField(9)
    validateForProxyless = _messages.BooleanField(10)