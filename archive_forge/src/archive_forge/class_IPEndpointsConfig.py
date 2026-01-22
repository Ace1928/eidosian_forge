from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IPEndpointsConfig(_messages.Message):
    """IP endpoints configuration.

  Fields:
    authorizedNetworksConfig: Configuration of authorized networks. If
      enabled, restricts access to the control plane based on source IP. It is
      invalid to specify both Cluster.masterAuthorizedNetworksConfig and this
      field at the same time.
    enablePublicEndpoint: Controls whether the control plane allows access
      through a public IP. It is invalid to specify both
      PrivateClusterConfig.enablePrivateEndpoint and this field at the same
      time.
    enabled: Controls whether to allow direct IP access.
    globalAccess: Controls whether the control plane's private endpoint is
      accessible from sources in other regions. It is invalid to specify both
      PrivateClusterMasterGlobalAccessConfig.enabled and this field at the
      same time.
    privateEndpoint: Output only. The internal IP address of this cluster's
      control plane. Only populated if enabled.
    publicEndpoint: Output only. The external IP address of this cluster's
      control plane. Only populated if enabled.
  """
    authorizedNetworksConfig = _messages.MessageField('MasterAuthorizedNetworksConfig', 1)
    enablePublicEndpoint = _messages.BooleanField(2)
    enabled = _messages.BooleanField(3)
    globalAccess = _messages.BooleanField(4)
    privateEndpoint = _messages.StringField(5)
    publicEndpoint = _messages.StringField(6)