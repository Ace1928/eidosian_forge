from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureControlPlane(_messages.Message):
    """AzureControlPlane represents the control plane configurations.

  Messages:
    TagsValue: Optional. A set of tags to apply to all underlying control
      plane Azure resources.

  Fields:
    configEncryption: Optional. Configuration related to vm config encryption.
    databaseEncryption: Optional. Configuration related to application-layer
      secrets encryption.
    endpointSubnetId: Optional. The ARM ID of the subnet where the control
      plane load balancer is deployed. When unspecified, it defaults to
      AzureControlPlane.subnet_id. Example: "/subscriptions/d00494d6-6f3c-
      4280-bbb2-
      899e163d1d30/resourceGroups/anthos_cluster_gkeust4/providers/Microsoft.N
      etwork/virtualNetworks/gke-vnet-gkeust4/subnets/subnetid123"
    mainVolume: Optional. Configuration related to the main volume provisioned
      for each control plane replica. The main volume is in charge of storing
      all of the cluster's etcd state. When unspecified, it defaults to a
      8-GiB Azure Disk.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    replicaPlacements: Optional. Configuration for where to place the control
      plane replicas. Up to three replica placement instances can be
      specified. If replica_placements is set, the replica placement instances
      will be applied to the three control plane replicas as evenly as
      possible.
    rootVolume: Optional. Configuration related to the root volume provisioned
      for each control plane replica. When unspecified, it defaults to 32-GiB
      Azure Disk.
    sshConfig: Required. SSH configuration for how to access the underlying
      control plane machines.
    subnetId: Optional. The ARM ID of the default subnet for the control
      plane. The control plane VMs are deployed in this subnet, unless
      `AzureControlPlane.replica_placements` is specified. This subnet will
      also be used as default for `AzureControlPlane.endpoint_subnet_id` if
      `AzureControlPlane.endpoint_subnet_id` is not specified. Similarly it
      will be used as default for
      `AzureClusterNetworking.service_load_balancer_subnet_id`. Example: `/sub
      scriptions//resourceGroups//providers/Microsoft.Network/virtualNetworks/
      /subnets/default`.
    tags: Optional. A set of tags to apply to all underlying control plane
      Azure resources.
    version: Required. The Kubernetes version to run on control plane replicas
      (e.g. `1.19.10-gke.1000`). You can list all supported versions on a
      given Google Cloud region by calling GetAzureServerConfig.
    vmSize: Optional. The Azure VM size name. Example: `Standard_DS2_v2`. For
      available VM sizes, see https://docs.microsoft.com/en-us/azure/virtual-
      machines/vm-naming-conventions. When unspecified, it defaults to
      `Standard_DS2_v2`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """Optional. A set of tags to apply to all underlying control plane Azure
    resources.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AzureConfigEncryption', 1)
    databaseEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AzureDatabaseEncryption', 2)
    endpointSubnetId = _messages.StringField(3)
    mainVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AzureDiskTemplate', 4)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AzureProxyConfig', 5)
    replicaPlacements = _messages.MessageField('GoogleCloudGkemulticloudV1ReplicaPlacement', 6, repeated=True)
    rootVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AzureDiskTemplate', 7)
    sshConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AzureSshConfig', 8)
    subnetId = _messages.StringField(9)
    tags = _messages.MessageField('TagsValue', 10)
    version = _messages.StringField(11)
    vmSize = _messages.StringField(12)