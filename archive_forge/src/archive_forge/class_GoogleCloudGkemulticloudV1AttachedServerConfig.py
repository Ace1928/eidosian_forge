from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedServerConfig(_messages.Message):
    """AttachedServerConfig provides information about supported Kubernetes
  versions

  Fields:
    name: The resource name of the config.
    validVersions: List of valid platform versions.
  """
    name = _messages.StringField(1)
    validVersions = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedPlatformVersionInfo', 2, repeated=True)