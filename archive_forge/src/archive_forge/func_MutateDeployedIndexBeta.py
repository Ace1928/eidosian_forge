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
def MutateDeployedIndexBeta(self, index_endpoint_ref, args):
    """Mutate a deployed index from an index endpoint."""
    automatic_resources = self.messages.GoogleCloudAiplatformV1beta1AutomaticResources()
    if args.min_replica_count is not None:
        automatic_resources.minReplicaCount = args.min_replica_count
    if args.max_replica_count is not None:
        automatic_resources.maxReplicaCount = args.max_replica_count
    deployed_index = self.messages.GoogleCloudAiplatformV1beta1DeployedIndex(automaticResources=automatic_resources, id=args.deployed_index_id, enableAccessLogging=args.enable_access_logging)
    if args.reserved_ip_ranges is not None:
        deployed_index.reservedIpRanges.extend(args.reserved_ip_ranges)
    if args.deployment_group is not None:
        deployed_index.deploymentGroup = args.deployment_group
    if args.audiences is not None and args.allowed_issuers is not None:
        auth_provider = self.messages.GoogleCloudAiplatformV1beta1DeployedIndexAuthConfigAuthProvider()
        auth_provider.audiences.extend(args.audiences)
        auth_provider.allowedIssuers.extend(args.allowed_issuers)
        deployed_index.deployedIndexAuthConfig = self.messages.GoogleCloudAiplatformV1beta1DeployedIndexAuthConfig(authProvider=auth_provider)
    request = self.messages.AiplatformProjectsLocationsIndexEndpointsMutateDeployedIndexRequest(indexEndpoint=index_endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1DeployedIndex=deployed_index)
    return self._service.MutateDeployedIndex(request)