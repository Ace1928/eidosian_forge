from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import flags
@base.Hidden
class GetLocations(base.Command):
    """Get all the replicas locations for a cloud spanner instance."""
    detailed_help = {'EXAMPLES': textwrap.dedent('        To get all replicas locations of a Cloud Spanner instance in this project, run:\n\n          $ {command} my-instance-id\n        ')}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    For `get-locations` command, we have one positional argument, `instanceId`
    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        flags.Instance().AddToParser(parser)
        parser.add_argument('--verbose', required=False, action='store_true', help='Indicates that both regions and types of replicas be returned.')
        parser.display_info.AddFormat('table(location:sort=1,type.if(verbose))')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. From `Args`, we extract command line
        arguments

    Returns:
      List of dict values for locations of instance
    """
        return instances.GetLocations(args.instance, args.verbose)