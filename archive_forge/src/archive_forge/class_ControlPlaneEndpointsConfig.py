from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlPlaneEndpointsConfig(_messages.Message):
    """Configuration for all of the cluster's control plane endpoints.

  Fields:
    dnsEndpointConfig: DNS endpoint configuration.
    enhancedIngress: Enhanced KCP ingress configuration.
    ipEndpointsConfig: IP endpoints configuration.
  """
    dnsEndpointConfig = _messages.MessageField('DNSEndpointConfig', 1)
    enhancedIngress = _messages.MessageField('EnhancedKCPIngress', 2)
    ipEndpointsConfig = _messages.MessageField('IPEndpointsConfig', 3)