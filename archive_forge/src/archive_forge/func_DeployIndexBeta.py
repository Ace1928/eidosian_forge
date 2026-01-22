from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def DeployIndexBeta(self, index_endpoint_ref, args):
    """Deploy an index to an index endpoint."""
    index_ref = _ParseIndex(args.index, args.region)
    deployed_index = self.messages.GoogleCloudAiplatformV1beta1DeployedIndex(displayName=args.display_name, id=args.deployed_index_id, index=index_ref.RelativeName())
    if args.reserved_ip_ranges is not None:
        deployed_index.reservedIpRanges.extend(args.reserved_ip_ranges)
    if args.deployment_group is not None:
        deployed_index.deploymentGroup = args.deployment_group
    if args.enable_access_logging is not None:
        deployed_index.enableAccessLogging = args.enable_access_logging
    if args.audiences is not None and args.allowed_issuers is not None:
        auth_provider = self.messages.GoogleCloudAiplatformV1beta1DeployedIndexAuthConfigAuthProvider()
        auth_provider.audiences.extend(args.audiences)
        auth_provider.allowedIssuers.extend(args.allowed_issuers)
        deployed_index.deployedIndexAuthConfig = self.messages.GoogleCloudAiplatformV1beta1DeployedIndexAuthConfig(authProvider=auth_provider)
    if args.machine_type is not None:
        dedicated_resources = self.messages.GoogleCloudAiplatformV1beta1DedicatedResources()
        dedicated_resources.machineSpec = self.messages.GoogleCloudAiplatformV1beta1MachineSpec(machineType=args.machine_type)
        if args.min_replica_count is not None:
            dedicated_resources.minReplicaCount = args.min_replica_count
        if args.max_replica_count is not None:
            dedicated_resources.maxReplicaCount = args.max_replica_count
        deployed_index.dedicatedResources = dedicated_resources
    else:
        automatic_resources = self.messages.GoogleCloudAiplatformV1beta1AutomaticResources()
        if args.min_replica_count is not None:
            automatic_resources.minReplicaCount = args.min_replica_count
        if args.max_replica_count is not None:
            automatic_resources.maxReplicaCount = args.max_replica_count
        deployed_index.automaticResources = automatic_resources
    deploy_index_req = self.messages.GoogleCloudAiplatformV1beta1DeployIndexRequest(deployedIndex=deployed_index)
    request = self.messages.AiplatformProjectsLocationsIndexEndpointsDeployIndexRequest(indexEndpoint=index_endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1DeployIndexRequest=deploy_index_req)
    return self._service.DeployIndex(request)