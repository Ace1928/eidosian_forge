from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedProxyConfig(_messages.Message):
    """Details of a proxy config.

  Fields:
    kubernetesSecret: The Kubernetes Secret resource that contains the HTTP(S)
      proxy configuration. The secret must be a JSON encoded proxy
      configuration as described in
  """
    kubernetesSecret = _messages.MessageField('GoogleCloudGkemulticloudV1KubernetesSecret', 1)