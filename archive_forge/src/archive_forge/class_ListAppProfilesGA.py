from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListAppProfilesGA(base.ListCommand):
    """List Bigtable app profiles."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To list all app profiles for an instance, run:\n\n            $ {command} --instance=my-instance-id\n\n          ')}

    @staticmethod
    def Args(parser):
        arguments.AddInstanceResourceArg(parser, 'to list app profiles for')
        parser.display_info.AddTransforms({'routingInfo': _TransformAppProfileToRoutingInfo})
        parser.display_info.AddFormat('\n          table(\n            name.basename():sort=1,\n            description:wrap,\n            routingInfo():wrap:label=ROUTING,\n            singleClusterRouting.allowTransactionalWrites.yesno("Yes"):label=TRANSACTIONAL_WRITES\n          )\n        ')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        instance_ref = args.CONCEPTS.instance.Parse()
        return app_profiles.List(instance_ref)