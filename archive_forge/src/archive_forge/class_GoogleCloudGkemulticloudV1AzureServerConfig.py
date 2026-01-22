from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureServerConfig(_messages.Message):
    """AzureServerConfig contains information about a Google Cloud location,
  such as supported Azure regions and Kubernetes versions.

  Fields:
    name: The `AzureServerConfig` resource name. `AzureServerConfig` names are
      formatted as `projects//locations//azureServerConfig`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    supportedAzureRegions: The list of supported Azure regions.
    validVersions: List of all released Kubernetes versions, including ones
      which are end of life and can no longer be used. Filter by the `enabled`
      property to limit to currently available versions. Valid versions
      supported for both create and update operations
  """
    name = _messages.StringField(1)
    supportedAzureRegions = _messages.StringField(2, repeated=True)
    validVersions = _messages.MessageField('GoogleCloudGkemulticloudV1AzureK8sVersionInfo', 3, repeated=True)