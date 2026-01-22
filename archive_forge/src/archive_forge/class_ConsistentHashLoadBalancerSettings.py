from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsistentHashLoadBalancerSettings(_messages.Message):
    """This message defines settings for a consistent hash style load balancer.

  Fields:
    httpCookie: Hash is based on HTTP Cookie. This field describes a HTTP
      cookie that will be used as the hash key for the consistent hash load
      balancer. If the cookie is not present, it will be generated. This field
      is applicable if the sessionAffinity is set to HTTP_COOKIE. Not
      supported when the backend service is referenced by a URL map that is
      bound to target gRPC proxy that has validateForProxyless field set to
      true.
    httpHeaderName: The hash based on the value of the specified header field.
      This field is applicable if the sessionAffinity is set to HEADER_FIELD.
    minimumRingSize: The minimum number of virtual nodes to use for the hash
      ring. Defaults to 1024. Larger ring sizes result in more granular load
      distributions. If the number of hosts in the load balancing pool is
      larger than the ring size, each host will be assigned a single virtual
      node.
  """
    httpCookie = _messages.MessageField('ConsistentHashLoadBalancerSettingsHttpCookie', 1)
    httpHeaderName = _messages.StringField(2)
    minimumRingSize = _messages.IntegerField(3)