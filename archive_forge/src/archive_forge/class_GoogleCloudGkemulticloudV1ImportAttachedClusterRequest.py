from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ImportAttachedClusterRequest(_messages.Message):
    """Request message for `AttachedClusters.ImportAttachedCluster` method.

  Fields:
    distribution: Required. The Kubernetes distribution of the underlying
      attached cluster. Supported values: ["eks", "aks"].
    fleetMembership: Required. The name of the fleet membership resource to
      import.
    platformVersion: Required. The platform version for the cluster (e.g.
      `1.19.0-gke.1000`). You can list all supported versions on a given
      Google Cloud region by calling GetAttachedServerConfig.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    validateOnly: If set, only validate the request, but do not actually
      import the cluster.
  """
    distribution = _messages.StringField(1)
    fleetMembership = _messages.StringField(2)
    platformVersion = _messages.StringField(3)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedProxyConfig', 4)
    validateOnly = _messages.BooleanField(5)