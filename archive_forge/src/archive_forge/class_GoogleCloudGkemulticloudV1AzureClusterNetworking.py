from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureClusterNetworking(_messages.Message):
    """ClusterNetworking contains cluster-wide networking configuration.

  Fields:
    podAddressCidrBlocks: Required. The IP address range of the pods in this
      cluster, in CIDR notation (e.g. `10.96.0.0/14`). All pods in the cluster
      get assigned a unique IPv4 address from these ranges. Only a single
      range is supported. This field cannot be changed after creation.
    serviceAddressCidrBlocks: Required. The IP address range for services in
      this cluster, in CIDR notation (e.g. `10.96.0.0/14`). All services in
      the cluster get assigned a unique IPv4 address from these ranges. Only a
      single range is supported. This field cannot be changed after creating a
      cluster.
    serviceLoadBalancerSubnetId: Optional. The ARM ID of the subnet where
      Kubernetes private service type load balancers are deployed. When
      unspecified, it defaults to AzureControlPlane.subnet_id. Example: "/subs
      criptions/d00494d6-6f3c-4280-bbb2-
      899e163d1d30/resourceGroups/anthos_cluster_gkeust4/providers/Microsoft.N
      etwork/virtualNetworks/gke-vnet-gkeust4/subnets/subnetid456"
    virtualNetworkId: Required. The Azure Resource Manager (ARM) ID of the
      VNet associated with your cluster. All components in the cluster (i.e.
      control plane and node pools) run on a single VNet. Example: `/subscript
      ions//resourceGroups//providers/Microsoft.Network/virtualNetworks/` This
      field cannot be changed after creation.
  """
    podAddressCidrBlocks = _messages.StringField(1, repeated=True)
    serviceAddressCidrBlocks = _messages.StringField(2, repeated=True)
    serviceLoadBalancerSubnetId = _messages.StringField(3)
    virtualNetworkId = _messages.StringField(4)