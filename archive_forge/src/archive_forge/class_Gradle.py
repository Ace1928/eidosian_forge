from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts.print_settings import settings_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Gradle(base.Command):
    """Print a snippet to add a repository to the Gradle build.gradle file."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '    To print a snippet for the repository set in the `artifacts/repository`\n    property in the default location:\n\n      $ {command}\n\n    To print a snippet for repository `my-repository` in the default location:\n\n      $ {command} --repository="my-repository"\n\n    To print a snippet using service account key:\n\n      $ {command} --json-key=path/to/key.json\n    '}

    @staticmethod
    def Args(parser):
        flags.GetRepoFlag().AddToParser(parser)
        flags.GetJsonKeyFlag('gradle').AddToParser(parser)
        parser.display_info.AddFormat('value(gradle)')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      A maven gradle snippet.
    """
        return {'gradle': settings_util.GetGradleSnippet(args)}