from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersPatchRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersPatchRequest object.

  Fields:
    googleCloudGkemulticloudV1AzureCluster: A
      GoogleCloudGkemulticloudV1AzureCluster resource to be passed as the
      request body.
    name: The name of this resource. Cluster names are formatted as
      `projects//locations//azureClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field can
      only include these fields from AzureCluster: * `description`. *
      `azureClient`. * `control_plane.version`. * `control_plane.vm_size`. *
      `annotations`. * `authorization.admin_users`. *
      `authorization.admin_groups`. * `control_plane.root_volume.size_gib`. *
      `azure_services_authentication`. *
      `azure_services_authentication.tenant_id`. *
      `azure_services_authentication.application_id`. *
      `control_plane.proxy_config`. *
      `control_plane.proxy_config.resource_group_id`. *
      `control_plane.proxy_config.secret_id`. *
      `control_plane.ssh_config.authorized_key`. *
      `logging_config.component_config.enable_components` *
      `monitoring_config.managed_prometheus_config.enabled`.
    validateOnly: If set, only validate the request, but do not actually
      update the cluster.
  """
    googleCloudGkemulticloudV1AzureCluster = _messages.MessageField('GoogleCloudGkemulticloudV1AzureCluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)