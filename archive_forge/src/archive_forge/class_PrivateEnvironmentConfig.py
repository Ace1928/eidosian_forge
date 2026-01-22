from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateEnvironmentConfig(_messages.Message):
    """The configuration information for configuring a Private IP Cloud
  Composer environment.

  Fields:
    cloudComposerConnectionSubnetwork: Optional. When specified, the
      environment will use Private Service Connect instead of VPC peerings to
      connect to Cloud SQL in the Tenant Project, and the PSC endpoint in the
      Customer Project will use an IP address from this subnetwork.
    cloudComposerNetworkIpv4CidrBlock: Optional. The CIDR block from which IP
      range for Cloud Composer Network in tenant project will be reserved.
      Needs to be disjoint from private_cluster_config.master_ipv4_cidr_block
      and cloud_sql_ipv4_cidr_block. This field is supported for Cloud
      Composer environments in versions composer-2.*.*-airflow-*.*.* and
      newer.
    cloudComposerNetworkIpv4ReservedRange: Output only. The IP range reserved
      for the tenant project's Cloud Composer network. This field is supported
      for Cloud Composer environments in versions composer-2.*.*-airflow-*.*.*
      and newer.
    cloudSqlIpv4CidrBlock: Optional. The CIDR block from which IP range in
      tenant project will be reserved for Cloud SQL. Needs to be disjoint from
      `web_server_ipv4_cidr_block`.
    enablePrivateBuildsOnly: Optional. If `true`, builds performed during
      operations that install Python packages have only private connectivity
      to Google services (including Artifact Registry) and VPC network (if
      either `NodeConfig.network` and `NodeConfig.subnetwork` fields or
      `NodeConfig.composer_network_attachment` field are specified). If
      `false`, the builds also have access to the internet. This field is
      supported for Cloud Composer environments in versions
      composer-3.*.*-airflow-*.*.* and newer.
    enablePrivateEnvironment: Optional. If `true`, a Private IP Cloud Composer
      environment is created. If this field is set to true,
      `IPAllocationPolicy.use_ip_aliases` must be set to true for Cloud
      Composer environments in versions composer-1.*.*-airflow-*.*.*.
    enablePrivatelyUsedPublicIps: Optional. When enabled, IPs from public
      (non-RFC1918) ranges can be used for
      `IPAllocationPolicy.cluster_ipv4_cidr_block` and
      `IPAllocationPolicy.service_ipv4_cidr_block`.
    networkingConfig: Optional. Configuration for the network connections
      configuration in the environment.
    privateClusterConfig: Optional. Configuration for the private GKE cluster
      for a Private IP Cloud Composer environment.
    webServerIpv4CidrBlock: Optional. The CIDR block from which IP range for
      web server will be reserved. Needs to be disjoint from
      `private_cluster_config.master_ipv4_cidr_block` and
      `cloud_sql_ipv4_cidr_block`. This field is supported for Cloud Composer
      environments in versions composer-1.*.*-airflow-*.*.*.
    webServerIpv4ReservedRange: Output only. The IP range reserved for the
      tenant project's App Engine VMs. This field is supported for Cloud
      Composer environments in versions composer-1.*.*-airflow-*.*.*.
  """
    cloudComposerConnectionSubnetwork = _messages.StringField(1)
    cloudComposerNetworkIpv4CidrBlock = _messages.StringField(2)
    cloudComposerNetworkIpv4ReservedRange = _messages.StringField(3)
    cloudSqlIpv4CidrBlock = _messages.StringField(4)
    enablePrivateBuildsOnly = _messages.BooleanField(5)
    enablePrivateEnvironment = _messages.BooleanField(6)
    enablePrivatelyUsedPublicIps = _messages.BooleanField(7)
    networkingConfig = _messages.MessageField('NetworkingConfig', 8)
    privateClusterConfig = _messages.MessageField('PrivateClusterConfig', 9)
    webServerIpv4CidrBlock = _messages.StringField(10)
    webServerIpv4ReservedRange = _messages.StringField(11)