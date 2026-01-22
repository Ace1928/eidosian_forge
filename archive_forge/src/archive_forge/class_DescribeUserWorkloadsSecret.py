import textwrap
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_secrets_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DescribeUserWorkloadsSecret(base.Command):
    """Get details about a user workloads Secret."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        base.Argument('secret_name', nargs='?', help='Name of the Secret.').AddToParser(parser)
        resource_args.AddEnvironmentResourceArg(parser, 'of the secret', positional=False)

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        return environments_user_workloads_secrets_util.GetUserWorkloadsSecret(env_resource, args.secret_name, release_track=self.ReleaseTrack())