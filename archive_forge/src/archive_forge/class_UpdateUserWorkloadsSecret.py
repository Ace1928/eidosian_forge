import textwrap
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_secrets_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UpdateUserWorkloadsSecret(base.Command):
    """Update a user workloads Secret."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'where the user workloads Secret must be updated', positional=False)
        parser.add_argument('--secret-file-path', type=str, help='Path to a local file with a single Kubernetes Secret in YAML format.', required=True)

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        response = environments_user_workloads_secrets_util.UpdateUserWorkloadsSecret(env_resource, args.secret_file_path, release_track=self.ReleaseTrack())
        log.status.Print(f'Secret {response.name} updated')