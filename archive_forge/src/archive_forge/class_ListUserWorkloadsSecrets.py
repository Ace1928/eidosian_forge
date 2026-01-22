import textwrap
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_secrets_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListUserWorkloadsSecrets(base.Command):
    """List user workloads Secrets."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'to list user workloads Secrets', positional=False)
        parser.display_info.AddFormat('table[box](name.segment(7),data)')

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        return environments_user_workloads_secrets_util.ListUserWorkloadsSecrets(env_resource, release_track=self.ReleaseTrack())