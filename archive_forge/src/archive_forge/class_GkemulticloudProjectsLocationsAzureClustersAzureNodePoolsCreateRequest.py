from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsCreateRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsCreateRequest
  object.

  Fields:
    azureNodePoolId: Required. A client provided ID the resource. Must be
      unique within the parent resource. The provided ID will be part of the
      AzureNodePool resource name formatted as
      `projects//locations//azureClusters//azureNodePools/`. Valid characters
      are `/a-z-/`. Cannot be longer than 63 characters.
    googleCloudGkemulticloudV1AzureNodePool: A
      GoogleCloudGkemulticloudV1AzureNodePool resource to be passed as the
      request body.
    parent: Required. The AzureCluster resource where this node pool will be
      created. `AzureCluster` names are formatted as
      `projects//locations//azureClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      create the node pool.
  """
    azureNodePoolId = _messages.StringField(1)
    googleCloudGkemulticloudV1AzureNodePool = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodePool', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)