from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteDestination(_messages.Message):
    """Specifications of a destination to which the request should be routed
  to.

  Fields:
    requestHeaderModifier: Optional. The specification for modifying the
      headers of a matching request prior to delivery of the request to the
      destination. If HeaderModifiers are set on both the Destination and the
      RouteAction, they will be merged. Conflicts between the two will not be
      resolved on the configuration.
    responseHeaderModifier: Optional. The specification for modifying the
      headers of a response prior to sending the response back to the client.
      If HeaderModifiers are set on both the Destination and the RouteAction,
      they will be merged. Conflicts between the two will not be resolved on
      the configuration.
    serviceName: The URL of a BackendService to route traffic to.
    weight: Specifies the proportion of requests forwarded to the backend
      referenced by the serviceName field. This is computed as: -
      weight/Sum(weights in this destination list). For non-zero values, there
      may be some epsilon from the exact proportion defined here depending on
      the precision an implementation supports. If only one serviceName is
      specified and it has a weight greater than 0, 100% of the traffic is
      forwarded to that backend. If weights are specified for any one service
      name, they need to be specified for all of them. If weights are
      unspecified for all services, then, traffic is distributed in equal
      proportions to all of them.
  """
    requestHeaderModifier = _messages.MessageField('HttpRouteHeaderModifier', 1)
    responseHeaderModifier = _messages.MessageField('HttpRouteHeaderModifier', 2)
    serviceName = _messages.StringField(3)
    weight = _messages.IntegerField(4, variant=_messages.Variant.INT32)