from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DisableVpcServiceControls(base.SilentCommand):
    """Disable VPC Service Controls for the peering connection."""
    detailed_help = {'DESCRIPTION': '          This command disables VPC Service Controls for the peering connection.\n\n          The local default route (destination 0.0.0.0/0, next hop default\n          internet gateway) is recreated in the service producer VPC network.\n          After the route is recreated, the service producer VPC network cannot\n          import a custom default route from the peering connection to the\n          customer VPC network.\n          ', 'EXAMPLES': '          To disable VPC Service Controls for a connection peering a network\n          called `my-network` on the current project to a service called\n          `your-service`, run:\n\n            $ {command} --network=my-network --service=your-service\n\n          To run the same command asynchronously (non-blocking), run:\n\n            $ {command} --network=my-network --service=your-service --async\n          '}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that can be used to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        parser.add_argument('--network', metavar='NETWORK', required=True, help='The network in the current project that is peered with the service.')
        parser.add_argument('--service', metavar='SERVICE', default='servicenetworking.googleapis.com', help='The service to enable VPC service controls for.')
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        """Run 'services vpc-peerings enable-vpc-service-controls'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.
    """
        project = properties.VALUES.core.project.Get(required=True)
        project_number = _GetProjectNumber(project)
        op = peering.DisableVpcServiceControls(project_number, args.service, args.network)
        if args.async_:
            cmd = OP_WAIT_CMD.format(op.name)
            log.status.Print('Asynchronous operation is in progress... Use the following command to wait for its completion:\n {0}'.format(cmd))
            return
        op = services_util.WaitOperation(op.name, peering.GetOperation)
        services_util.PrintOperation(op)