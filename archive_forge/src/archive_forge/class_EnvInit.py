from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import spanner_util
from googlecloudsdk.command_lib.emulators import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class EnvInit(base.Command):
    """Print the commands required to export Spanner emulator's env variables."""
    detailed_help = {'EXAMPLES': '          To print the env variables exports for a Spanner emulator, run:\n\n            $ {command}\n          '}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('config[export]')

    def Run(self, args):
        data_dir = spanner_util.GetDataDir()
        return util.ReadEnvYaml(data_dir)