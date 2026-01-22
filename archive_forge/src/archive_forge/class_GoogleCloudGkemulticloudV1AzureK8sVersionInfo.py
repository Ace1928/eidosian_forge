from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureK8sVersionInfo(_messages.Message):
    """Kubernetes version information of GKE cluster on Azure.

  Fields:
    enabled: Optional. True if the version is available for cluster creation.
      If a version is enabled for creation, it can be used to create new
      clusters. Otherwise, cluster creation will fail. However, cluster
      upgrade operations may succeed, even if the version is not enabled.
    endOfLife: Optional. True if this cluster version belongs to a minor
      version that has reached its end of life and is no longer in scope to
      receive security and bug fixes.
    endOfLifeDate: Optional. The estimated date (in Pacific Time) when this
      cluster version will reach its end of life. Or if this version is no
      longer supported (the `end_of_life` field is true), this is the actual
      date (in Pacific time) when the version reached its end of life.
    releaseDate: Optional. The date (in Pacific Time) when the cluster version
      was released.
    version: Kubernetes version name (for example, `1.19.10-gke.1000`)
  """
    enabled = _messages.BooleanField(1)
    endOfLife = _messages.BooleanField(2)
    endOfLifeDate = _messages.MessageField('GoogleTypeDate', 3)
    releaseDate = _messages.MessageField('GoogleTypeDate', 4)
    version = _messages.StringField(5)