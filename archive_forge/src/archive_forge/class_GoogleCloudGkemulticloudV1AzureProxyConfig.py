from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureProxyConfig(_messages.Message):
    """Details of a proxy config stored in Azure Key Vault.

  Fields:
    resourceGroupId: The ARM ID the of the resource group containing proxy
      keyvault. Resource group ids are formatted as
      `/subscriptions//resourceGroups/`.
    secretId: The URL the of the proxy setting secret with its version. The
      secret must be a JSON encoded proxy configuration as described in
      https://cloud.google.com/anthos/clusters/docs/multi-cloud/azure/how-
      to/use-a-proxy#create_a_proxy_configuration_file Secret ids are
      formatted as `https://.vault.azure.net/secrets//`.
  """
    resourceGroupId = _messages.StringField(1)
    secretId = _messages.StringField(2)