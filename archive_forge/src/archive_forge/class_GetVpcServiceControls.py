from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetVpcServiceControls(base.DescribeCommand):
    """Get VPC state of Service Controls for the peering connection."""
    detailed_help = {'DESCRIPTION': '        This command provides the state of the VPC Service Controls for a\n        connection.  The state can be enabled or disabled.\n\n        When enabled, Google Cloud makes the following route configuration\n        changes in the service producer VPC network: Google Cloud removes the\n        IPv4 default route (destination 0.0.0.0/0, next hop default internet\n        gateway), Google Cloud then creates an IPv4 route for destination\n        199.36.153.4/30 using the default internet gateway next hop.\n\n        When enabled, Google Cloud also creates Cloud DNS managed private\n        zones and authorizes those zones for the service producer VPC network.\n        The zones include googleapis.com, pkg.dev, gcr.io, and other necessary\n        domains or host names for Google APIs and services that are compatible\n        with VPC Service Controls. Record data in the zones resolves all host\n        names to 199.36.153.4, 199.36.153.5, 199.36.153.6, and 199.36.153.7.\n\n        When disabled, Google Cloud makes the following route configuration\n        changes in the service producer VPC network: Google Cloud restores a\n        default route (destination 0.0.0.0/0, next hop default internet\n        gateway), Google Cloud also deletes the Cloud DNS managed private\n        zones that provided the host name overrides.\n\n        While enabled, the service producer VPC network can still import\n        static and dynamic routes from the peered customer network if you\n        enable custom route export. These custom routes can include a default\n        route. For this reason, this command is not to be used solely as a\n        means for preventing access to the internet.\n        ', 'EXAMPLES': '        To get the status of the VPC Service Controls for a connection peering\n        a network called `my-network` on the current project to a service called\n        `your-service`, run:\n\n          $ {command} --network=my-network --service=your-service\n        '}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that can be used to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        parser.add_argument('--network', metavar='NETWORK', required=True, help='The network in the current project that is peered with the service.')
        parser.add_argument('--service', metavar='SERVICE', default='servicenetworking.googleapis.com', help='The service to get VPC service controls for.')

    def Run(self, args):
        """Run 'services vpc-peerings get-vpc-service-controls'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The state of the Vpc Service Controls, that is enabled or disabled.
    """
        project = properties.VALUES.core.project.Get(required=True)
        project_number = projects_util.GetProjectNumber(project)
        return peering.GetVpcServiceControls(project_number, args.service, args.network)