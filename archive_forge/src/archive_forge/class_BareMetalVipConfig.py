from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalVipConfig(_messages.Message):
    """Specifies the VIP config for the bare metal load balancer.

  Fields:
    controlPlaneVip: The VIP which you previously set aside for the Kubernetes
      API of this bare metal user cluster.
    ingressVip: The VIP which you previously set aside for ingress traffic
      into this bare metal user cluster.
  """
    controlPlaneVip = _messages.StringField(1)
    ingressVip = _messages.StringField(2)