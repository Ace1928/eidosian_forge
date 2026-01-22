from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CircuitBreakers(_messages.Message):
    """Settings controlling the volume of requests, connections and retries to
  this backend service.

  Fields:
    connectTimeout: The timeout for new network connections to hosts.
    maxConnections: The maximum number of connections to the backend service.
      If not specified, there is no limit. Not supported when the backend
      service is referenced by a URL map that is bound to target gRPC proxy
      that has validateForProxyless field set to true.
    maxPendingRequests: The maximum number of pending requests allowed to the
      backend service. If not specified, there is no limit. Not supported when
      the backend service is referenced by a URL map that is bound to target
      gRPC proxy that has validateForProxyless field set to true.
    maxRequests: The maximum number of parallel requests that allowed to the
      backend service. If not specified, there is no limit.
    maxRequestsPerConnection: Maximum requests for a single connection to the
      backend service. This parameter is respected by both the HTTP/1.1 and
      HTTP/2 implementations. If not specified, there is no limit. Setting
      this parameter to 1 will effectively disable keep alive. Not supported
      when the backend service is referenced by a URL map that is bound to
      target gRPC proxy that has validateForProxyless field set to true.
    maxRetries: The maximum number of parallel retries allowed to the backend
      cluster. If not specified, the default is 1. Not supported when the
      backend service is referenced by a URL map that is bound to target gRPC
      proxy that has validateForProxyless field set to true.
  """
    connectTimeout = _messages.MessageField('Duration', 1)
    maxConnections = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    maxPendingRequests = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maxRequests = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    maxRequestsPerConnection = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    maxRetries = _messages.IntegerField(6, variant=_messages.Variant.INT32)