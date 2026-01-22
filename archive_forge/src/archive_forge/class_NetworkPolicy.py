from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkPolicy(_messages.Message):
    """Represents a network policy resource. Network policies are regional
  resources. You can use a network policy to enable or disable internet access
  and external IP access. Network policies are associated with a VMware Engine
  network, which might span across regions. For a given region, a network
  policy applies to all private clouds in the VMware Engine network associated
  with the policy.

  Fields:
    createTime: Output only. Creation time of this resource.
    description: Optional. User-provided description for this network policy.
    edgeServicesCidr: Required. IP address range in CIDR notation used to
      create internet access and external IP access. An RFC 1918 CIDR block,
      with a "/26" prefix, is required. The range cannot overlap with any
      prefixes either in the consumer VPC network or in use by the private
      clouds attached to that VPC network.
    externalIp: Network service that allows External IP addresses to be
      assigned to VMware workloads. This service can only be enabled when
      `internet_access` is also enabled.
    internetAccess: Network service that allows VMware workloads to access the
      internet.
    name: Output only. The resource name of this network policy. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-network-
      policy`
    network: Optional. Deprecated: Optional. Name of the network in the
      consumer project which is peered or will be peered with the service
      network. Provide the network name in the form of
      `projects/{project}/global/networks/{network}`, where `{project}` is the
      project ID or project number of the project containing the network. In
      case of shared VPC, use the project ID or project number of the host
      project containing the shared VPC network.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    vmwareEngineNetwork: Optional. The relative resource name of the VMware
      Engine network. Specify the name in the following form: `projects/{proje
      ct}/locations/{location}/vmwareEngineNetworks/{vmware_engine_network_id}
      ` where `{project}` can either be a project number or a project ID.
    vmwareEngineNetworkCanonical: Output only. The canonical name of the
      VMware Engine network in the form: `projects/{project_number}/locations/
      {location}/vmwareEngineNetworks/{vmware_engine_network_id}`
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    edgeServicesCidr = _messages.StringField(3)
    externalIp = _messages.MessageField('NetworkService', 4)
    internetAccess = _messages.MessageField('NetworkService', 5)
    name = _messages.StringField(6)
    network = _messages.StringField(7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    vmwareEngineNetwork = _messages.StringField(10)
    vmwareEngineNetworkCanonical = _messages.StringField(11)