from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.index_endpoints import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import index_endpoints_util
from googlecloudsdk.command_lib.ai import operations_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UndeployIndexV1(base.Command):
    """Undeploy an index from a Vertex AI index endpoint.

  ## EXAMPLES

  To undeploy the deployed-index ``deployed-index-345'' from an index endpoint
  ``456'' under project ``example'' in region ``us-central1'', run:

    $ {command} 456 --project=example --region=us-central1
    --deployed-index-id=deployed-index-345
  """

    @staticmethod
    def Args(parser):
        flags.AddIndexEndpointResourceArg(parser, 'to undeploy an index')
        flags.GetDeployedIndexId().AddToParser(parser)

    def _Run(self, args, version):
        index_endpoint_ref = args.CONCEPTS.index_endpoint.Parse()
        region = index_endpoint_ref.AsDict()['locationsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            index_endpoint_client = client.IndexEndpointsClient(version=version)
            if version == constants.GA_VERSION:
                operation = index_endpoint_client.UndeployIndex(index_endpoint_ref, args)
            else:
                operation = index_endpoint_client.UndeployIndexBeta(index_endpoint_ref, args)
            return operations_util.WaitForOpMaybe(operations_client=operations.OperationsClient(version=version), op=operation, op_ref=index_endpoints_util.ParseIndexEndpointOperation(operation.name))

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)