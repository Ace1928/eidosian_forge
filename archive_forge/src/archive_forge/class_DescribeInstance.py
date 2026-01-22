from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class DescribeInstance(base.DescribeCommand):
    """Describe an existing Bigtable instance."""
    detailed_help = {'EXAMPLES': textwrap.dedent("          To view an instance's description, run:\n\n            $ {command} my-instance-id\n\n          ")}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddInstanceResourceArg(parser, 'to describe', positional=True)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        ref = resources.REGISTRY.Parse(args.instance, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='bigtableadmin.projects.instances')
        msg = util.GetAdminMessages().BigtableadminProjectsInstancesGetRequest(name=ref.RelativeName())
        instance = cli.projects_instances.Get(msg)
        return instance