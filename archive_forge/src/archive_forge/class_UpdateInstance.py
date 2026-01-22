from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util as bigtable_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
class UpdateInstance(base.UpdateCommand):
    """Modify an existing Bigtable instance."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To update the display name for an instance, run:\n\n            $ {command} my-instance-id --display-name="Updated Instance Name"\n\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.ArgAdder(parser).AddInstanceDisplayName()
        arguments.AddInstanceResourceArg(parser, 'to update', positional=True)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cli = bigtable_util.GetAdminClient()
        ref = bigtable_util.GetInstanceRef(args.instance)
        msgs = bigtable_util.GetAdminMessages()
        instance = cli.projects_instances.Get(msgs.BigtableadminProjectsInstancesGetRequest(name=ref.RelativeName()))
        instance.state = None
        if args.display_name:
            instance.displayName = args.display_name
        instance = cli.projects_instances.Update(instance)
        log.UpdatedResource(instance.name, kind='instance')
        return instance