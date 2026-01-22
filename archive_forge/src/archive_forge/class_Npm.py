from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts.print_settings import settings_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Npm(base.Command):
    """Print credential settings to add to the .npmrc file.

  Print credential settings to add to the .npmrc file for connecting to an npm
  repository.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '    To print a snippet for the repository set in the `artifacts/repository`\n    property in the default location:\n\n      $ {command}\n\n    To print a snippet for repository `my-repository` in the default location:\n\n      $ {command} --repository="my-repository"\n\n    To print a snippet using service account key:\n\n      $ {command} --json-key=path/to/key.json\n\n    To print a snippet for the repository set in the `artifacts/repository`\n    property with scope @my-company:\n\n      $ {command} --scope=@my-company\n    '}

    @staticmethod
    def Args(parser):
        flags.GetRepoFlag().AddToParser(parser)
        flags.GetJsonKeyFlag('npm').AddToParser(parser)
        flags.GetScopeFlag().AddToParser(parser)
        parser.display_info.AddFormat('value(npm)')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      An npm settings snippet.
    """
        return {'npm': settings_util.GetNpmSettingsSnippet(args)}