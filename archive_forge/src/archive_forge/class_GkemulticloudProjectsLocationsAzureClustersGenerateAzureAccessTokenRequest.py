from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersGenerateAzureAccessTokenRequest(_messages.Message):
    """A
  GkemulticloudProjectsLocationsAzureClustersGenerateAzureAccessTokenRequest
  object.

  Fields:
    azureCluster: Required. The name of the AzureCluster resource to
      authenticate to. `AzureCluster` names are formatted as
      `projects//locations//azureClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    azureCluster = _messages.StringField(1, required=True)