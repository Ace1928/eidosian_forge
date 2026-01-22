from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import deployment_resource_pools_util
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core.console import console_io
def _RunBeta(args):
    """Delete a Vertex AI deployment resource pool."""
    version = constants.BETA_VERSION
    deployment_resource_pool_ref = args.CONCEPTS.deployment_resource_pool.Parse()
    args.region = deployment_resource_pool_ref.AsDict()['locationsId']
    deployment_resource_pool_id = deployment_resource_pool_ref.AsDict()['deploymentResourcePoolsId']
    with endpoint_util.AiplatformEndpointOverrides(version, region=args.region):
        deployment_resource_pools_client = client.DeploymentResourcePoolsClient(version=version)
        operation_client = operations.OperationsClient()
        console_io.PromptContinue('This will delete deployment resource pool [{}]...'.format(deployment_resource_pool_id), cancel_on_no=True)
        op = deployment_resource_pools_client.DeleteBeta(deployment_resource_pool_ref)
        return operations_util.WaitForOpMaybe(operation_client, op, deployment_resource_pools_util.ParseOperation(op.name), log_method='delete')