from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeployedIndex(_messages.Message):
    """A deployment of an Index. IndexEndpoints contain one or more
  DeployedIndexes.

  Fields:
    automaticResources: Optional. A description of resources that the
      DeployedIndex uses, which to large degree are decided by Vertex AI, and
      optionally allows only a modest additional configuration. If
      min_replica_count is not set, the default value is 2 (we don't provide
      SLA when min_replica_count=1). If max_replica_count is not set, the
      default value is min_replica_count. The max allowed replica count is
      1000.
    createTime: Output only. Timestamp when the DeployedIndex was created.
    dedicatedResources: Optional. A description of resources that are
      dedicated to the DeployedIndex, and that need a higher degree of manual
      configuration. The field min_replica_count must be set to a value
      strictly greater than 0, or else validation will fail. We don't provide
      SLA when min_replica_count=1. If max_replica_count is not set, the
      default value is min_replica_count. The max allowed replica count is
      1000. Available machine types for SMALL shard: e2-standard-2 and all
      machine types available for MEDIUM and LARGE shard. Available machine
      types for MEDIUM shard: e2-standard-16 and all machine types available
      for LARGE shard. Available machine types for LARGE shard: e2-highmem-16,
      n2d-standard-32. n1-standard-16 and n1-standard-32 are still available,
      but we recommend e2-standard-16 and e2-highmem-16 for cost efficiency.
    deployedIndexAuthConfig: Optional. If set, the authentication is enabled
      for the private endpoint.
    deploymentGroup: Optional. The deployment group can be no longer than 64
      characters (eg: 'test', 'prod'). If not set, we will use the 'default'
      deployment group. Creating `deployment_groups` with `reserved_ip_ranges`
      is a recommended practice when the peered network has multiple peering
      ranges. This creates your deployments from predictable IP spaces for
      easier traffic administration. Also, one deployment_group (except
      'default') can only be used with the same reserved_ip_ranges which means
      if the deployment_group has been used with reserved_ip_ranges: [a, b,
      c], using it with [a, b] or [d, e] is disallowed. Note: we only support
      up to 5 deployment groups(not including 'default').
    displayName: The display name of the DeployedIndex. If not provided upon
      creation, the Index's display_name is used.
    enableAccessLogging: Optional. If true, private endpoint's access logs are
      sent to Cloud Logging. These logs are like standard server access logs,
      containing information like timestamp and latency for each MatchRequest.
      Note that logs may incur a cost, especially if the deployed index
      receives a high queries per second rate (QPS). Estimate your costs
      before enabling this option.
    id: Required. The user specified ID of the DeployedIndex. The ID can be up
      to 128 characters long and must start with a letter and only contain
      letters, numbers, and underscores. The ID must be unique within the
      project it is created in.
    index: Required. The name of the Index this is the deployment of. We may
      refer to this Index as the DeployedIndex's "original" Index.
    indexSyncTime: Output only. The DeployedIndex may depend on various data
      on its original Index. Additionally when certain changes to the original
      Index are being done (e.g. when what the Index contains is being
      changed) the DeployedIndex may be asynchronously updated in the
      background to reflect these changes. If this timestamp's value is at
      least the Index.update_time of the original Index, it means that this
      DeployedIndex and the original Index are in sync. If this timestamp is
      older, then to see which updates this DeployedIndex already contains
      (and which it does not), one must list the operations that are running
      on the original Index. Only the successfully completed Operations with
      update_time equal or before this sync time are contained in this
      DeployedIndex.
    privateEndpoints: Output only. Provides paths for users to send requests
      directly to the deployed index services running on Cloud via private
      services access. This field is populated if network is configured.
    reservedIpRanges: Optional. A list of reserved ip ranges under the VPC
      network that can be used for this DeployedIndex. If set, we will deploy
      the index within the provided ip ranges. Otherwise, the index might be
      deployed to any ip ranges under the provided VPC network. The value
      should be the name of the address
      (https://cloud.google.com/compute/docs/reference/rest/v1/addresses)
      Example: ['vertex-ai-ip-range']. For more information about subnets and
      network IP ranges, please see https://cloud.google.com/vpc/docs/subnets#
      manually_created_subnet_ip_ranges.
  """
    automaticResources = _messages.MessageField('GoogleCloudAiplatformV1beta1AutomaticResources', 1)
    createTime = _messages.StringField(2)
    dedicatedResources = _messages.MessageField('GoogleCloudAiplatformV1beta1DedicatedResources', 3)
    deployedIndexAuthConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1DeployedIndexAuthConfig', 4)
    deploymentGroup = _messages.StringField(5)
    displayName = _messages.StringField(6)
    enableAccessLogging = _messages.BooleanField(7)
    id = _messages.StringField(8)
    index = _messages.StringField(9)
    indexSyncTime = _messages.StringField(10)
    privateEndpoints = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexPrivateEndpoints', 11)
    reservedIpRanges = _messages.StringField(12, repeated=True)