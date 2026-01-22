from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class EnvUnset(base.Command):
    """Print the commands required to unset a datastore emulators env variables.
  """
    detailed_help = {'EXAMPLES': '\nTo print the commands necessary to unset the env variables for\na datastore emulator, run:\n\n  $ {command} --data-dir=DATA-DIR\n'}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('config[unset]')

    def Run(self, args):
        return util.ReadEnvYaml(args.data_dir)