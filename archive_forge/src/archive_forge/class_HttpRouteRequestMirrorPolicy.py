from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteRequestMirrorPolicy(_messages.Message):
    """Specifies the policy on how requests are shadowed to a separate mirrored
  destination service. The proxy does not wait for responses from the shadow
  service. Prior to sending traffic to the shadow service, the host/authority
  header is suffixed with -shadow.

  Fields:
    destination: The destination the requests will be mirrored to. The weight
      of the destination will be ignored.
    mirrorPercent: Optional. The percentage of requests to get mirrored to the
      desired destination.
  """
    destination = _messages.MessageField('HttpRouteDestination', 1)
    mirrorPercent = _messages.FloatField(2, variant=_messages.Variant.FLOAT)