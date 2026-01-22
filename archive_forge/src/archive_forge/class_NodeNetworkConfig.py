from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeNetworkConfig(_messages.Message):
    """Parameters for node pool-level network config.

  Fields:
    additionalNodeNetworkConfigs: We specify the additional node networks for
      this node pool using this list. Each node network corresponds to an
      additional interface
    additionalPodNetworkConfigs: We specify the additional pod networks for
      this node pool using this list. Each pod network corresponds to an
      additional alias IP range for the node
    createPodRange: Input only. Whether to create a new range for pod IPs in
      this node pool. Defaults are provided for `pod_range` and
      `pod_ipv4_cidr_block` if they are not specified. If neither
      `create_pod_range` or `pod_range` are specified, the cluster-level
      default (`ip_allocation_policy.cluster_ipv4_cidr_block`) is used. Only
      applicable if `ip_allocation_policy.use_ip_aliases` is true. This field
      cannot be changed after the node pool has been created.
    enablePrivateNodes: Whether nodes have internal IP addresses only. If
      enable_private_nodes is not specified, then the value is derived from
      cluster.privateClusterConfig.enablePrivateNodes
    networkPerformanceConfig: Network bandwidth tier configuration.
    podCidrOverprovisionConfig: [PRIVATE FIELD] Pod CIDR size overprovisioning
      config for the nodepool. Pod CIDR size per node depends on
      max_pods_per_node. By default, the value of max_pods_per_node is rounded
      off to next power of 2 and we then double that to get the size of pod
      CIDR block per node. Example: max_pods_per_node of 30 would result in 64
      IPs (/26). This config can disable the doubling of IPs (we still round
      off to next power of 2) Example: max_pods_per_node of 30 will result in
      32 IPs (/27) when overprovisioning is disabled.
    podIpv4CidrBlock: The IP address range for pod IPs in this node pool. Only
      applicable if `create_pod_range` is true. Set to blank to have a range
      chosen with the default size. Set to /netmask (e.g. `/14`) to have a
      range chosen with a specific netmask. Set to a
      [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
      notation (e.g. `10.96.0.0/14`) to pick a specific range to use. Only
      applicable if `ip_allocation_policy.use_ip_aliases` is true. This field
      cannot be changed after the node pool has been created.
    podIpv4RangeUtilization: Output only. [Output only] The utilization of the
      IPv4 range for the pod. The ratio is Usage/[Total number of IPs in the
      secondary range], Usage=numNodes*numZones*podIPsPerNode.
    podRange: The ID of the secondary range for pod IPs. If `create_pod_range`
      is true, this ID is used for the new range. If `create_pod_range` is
      false, uses an existing secondary range with this ID. Only applicable if
      `ip_allocation_policy.use_ip_aliases` is true. This field cannot be
      changed after the node pool has been created.
  """
    additionalNodeNetworkConfigs = _messages.MessageField('AdditionalNodeNetworkConfig', 1, repeated=True)
    additionalPodNetworkConfigs = _messages.MessageField('AdditionalPodNetworkConfig', 2, repeated=True)
    createPodRange = _messages.BooleanField(3)
    enablePrivateNodes = _messages.BooleanField(4)
    networkPerformanceConfig = _messages.MessageField('NetworkPerformanceConfig', 5)
    podCidrOverprovisionConfig = _messages.MessageField('PodCIDROverprovisionConfig', 6)
    podIpv4CidrBlock = _messages.StringField(7)
    podIpv4RangeUtilization = _messages.FloatField(8)
    podRange = _messages.StringField(9)