from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyHealthCheckTargets(_messages.Message):
    """HealthCheckTargets describes endpoints to health-check when responding
  to Routing Policy queries. Only the healthy endpoints will be included in
  the response. Only one of internal_load_balancer and external_endpoints
  should be set.

  Fields:
    externalEndpoints: The Internet IP addresses to be health checked. The
      format matches the format of ResourceRecordSet.rrdata as defined in RFC
      1035 (section 5) and RFC 1034 (section 3.6.1)
    internalLoadBalancers: Configuration for internal load balancers to be
      health checked.
  """
    externalEndpoints = _messages.StringField(1, repeated=True)
    internalLoadBalancers = _messages.MessageField('RRSetRoutingPolicyLoadBalancerTarget', 2, repeated=True)