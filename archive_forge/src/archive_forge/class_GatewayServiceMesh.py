from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayServiceMesh(_messages.Message):
    """Information about the Kubernetes Gateway API service mesh configuration.

  Fields:
    deployment: Required. Name of the Kubernetes Deployment whose traffic is
      managed by the specified HTTPRoute and Service.
    httpRoute: Required. Name of the Gateway API HTTPRoute.
    routeUpdateWaitTime: Optional. The time to wait for route updates to
      propagate. The maximum configurable time is 3 hours, in seconds format.
      If unspecified, there is no wait time.
    service: Required. Name of the Kubernetes Service.
    stableCutbackDuration: Optional. The amount of time to migrate traffic
      back from the canary Service to the original Service during the stable
      phase deployment. If specified, must be between 15s and 3600s. If
      unspecified, there is no cutback time.
  """
    deployment = _messages.StringField(1)
    httpRoute = _messages.StringField(2)
    routeUpdateWaitTime = _messages.StringField(3)
    service = _messages.StringField(4)
    stableCutbackDuration = _messages.StringField(5)