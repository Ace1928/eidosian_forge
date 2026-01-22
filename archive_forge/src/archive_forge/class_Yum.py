from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts.print_settings import settings_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Yum(base.Command):
    """Print settings to add to the yum.repos.d directory.

  Print settings to add to the yum.repos.d directory for connecting to a Yum
  repository.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '    To print a snippet for the repository set in the `artifacts/repository`\n    property in the default location:\n\n      $ {command}\n\n    To print a snippet for repository `my-repository` in the default location:\n\n      $ {command} --repository="my-repository"\n    '}

    @staticmethod
    def Args(parser):
        flags.GetRepoFlag().AddToParser(parser)
        parser.display_info.AddFormat('value(yum)')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      A Yum settings snippet.
    """
        return {'yum': settings_util.GetYumSettingsSnippet(args)}