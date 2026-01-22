from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class WaitBeta(base.Command):
    """Poll an operation for completion."""
    detailed_help = {'EXAMPLES': _EXAMPLES}

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        """Registers flags for this command."""
        cluster_flags.AddOperationResourceArg(parser, 'to wait for completion')

    def Run(self, args):
        """Runs the wait command."""
        operation_client = operations.OperationsClient()
        operation_ref = args.CONCEPTS.operation_id.Parse()
        return operation_client.Wait(operation_ref=operation_ref)