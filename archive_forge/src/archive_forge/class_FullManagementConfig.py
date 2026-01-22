from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FullManagementConfig(_messages.Message):
    """Configuration of the full (Autopilot) cluster management

  Fields:
    clusterCidrBlock: Optional. The IP address range for the cluster pod IPs.
      Set to blank to have a range chosen with the default size. Set to
      /netmask (e.g. /14) to have a range chosen with a specific netmask. Set
      to a CIDR notation (e.g. 10.96.0.0/14) from the RFC-1918 private
      networks (e.g. 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) to pick a
      specific range to use.
    clusterNamedRange: Optional. The name of the existing secondary range in
      the cluster's subnetwork to use for pod IP addresses. Alternatively,
      cluster_cidr_block can be used to automatically create a GKE-managed
      one.
    masterAuthorizedNetworksConfig: Optional. Master Authorized Network that
      supports multiple CIDR blocks. Allows access to the k8s master from
      multiple blocks. It cannot be set at the same time with the field
      man_block.
    masterIpv4CidrBlock: Optional. The /28 network that the masters will use.
    network: Optional. Name of the VPC Network to put the GKE cluster and
      nodes in. The VPC will be created if it doesn't exist.
    servicesCidrBlock: Optional. The IP address range for the cluster service
      IPs. Set to blank to have a range chosen with the default size. Set to
      /netmask (e.g. /14) to have a range chosen with a specific netmask. Set
      to a CIDR notation (e.g. 10.96.0.0/14) from the RFC-1918 private
      networks (e.g. 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) to pick a
      specific range to use.
    servicesNamedRange: Optional. The name of the existing secondary range in
      the cluster's subnetwork to use for service ClusterIPs. Alternatively,
      services_cidr_block can be used to automatically create a GKE-managed
      one.
    subnet: Optional. Specifies the subnet that the interface will be part of.
      Network key must be specified and the subnet must be a subnetwork of the
      specified network.
  """
    clusterCidrBlock = _messages.StringField(1)
    clusterNamedRange = _messages.StringField(2)
    masterAuthorizedNetworksConfig = _messages.MessageField('MasterAuthorizedNetworksConfig', 3)
    masterIpv4CidrBlock = _messages.StringField(4)
    network = _messages.StringField(5)
    servicesCidrBlock = _messages.StringField(6)
    servicesNamedRange = _messages.StringField(7)
    subnet = _messages.StringField(8)