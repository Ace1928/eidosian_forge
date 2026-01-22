from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressPolicy(_messages.Message):
    """Policy for ingress into ServicePerimeter. IngressPolicies match requests
  based on `ingress_from` and `ingress_to` stanzas. For an ingress policy to
  match, both the `ingress_from` and `ingress_to` stanzas must be matched. If
  an IngressPolicy matches a request, the request is allowed through the
  perimeter boundary from outside the perimeter. For example, access from the
  internet can be allowed either based on an AccessLevel or, for traffic
  hosted on Google Cloud, the project of the source network. For access from
  private networks, using the project of the hosting network is required.
  Individual ingress policies can be limited by restricting which services
  and/or actions they match using the `ingress_to` field.

  Fields:
    ingressFrom: Defines the conditions on the source of a request causing
      this IngressPolicy to apply.
    ingressTo: Defines the conditions on the ApiOperation and request
      destination that cause this IngressPolicy to apply.
  """
    ingressFrom = _messages.MessageField('IngressFrom', 1)
    ingressTo = _messages.MessageField('IngressTo', 2)