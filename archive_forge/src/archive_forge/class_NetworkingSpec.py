from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NetworkingSpec(_messages.Message):
    """NetworkingSpec defines the desired state of Networking

  Fields:
    enabled: A boolean attribute.
    loadbalancertype: LoadBalancerType is whether the istio ingress is
      internal or external possible values are internal | external(implicit).
  """
    enabled = _messages.BooleanField(1)
    loadbalancertype = _messages.StringField(2)