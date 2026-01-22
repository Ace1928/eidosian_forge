from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteDestination(_messages.Message):
    """The destination to which traffic will be routed.

  Fields:
    serviceName: Required. The URL of a destination service to which to route
      traffic. Must refer to either a BackendService or
      ServiceDirectoryService.
    weight: Optional. Specifies the proportion of requests forwarded to the
      backend referenced by the serviceName field. This is computed as: -
      weight/Sum(weights in this destination list). For non-zero values, there
      may be some epsilon from the exact proportion defined here depending on
      the precision an implementation supports. If only one serviceName is
      specified and it has a weight greater than 0, 100% of the traffic is
      forwarded to that backend. If weights are specified for any one service
      name, they need to be specified for all of them. If weights are
      unspecified for all services, then, traffic is distributed in equal
      proportions to all of them.
  """
    serviceName = _messages.StringField(1)
    weight = _messages.IntegerField(2, variant=_messages.Variant.INT32)