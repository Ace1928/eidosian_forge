import textwrap
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_config_maps_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DeleteUserWorkloadsConfigMap(base.Command):
    """Delete a user workloads ConfigMap."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        base.Argument('config_map_name', nargs='?', help='Name of the ConfigMap.').AddToParser(parser)
        resource_args.AddEnvironmentResourceArg(parser, 'of the config_map', positional=False)

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        environments_user_workloads_config_maps_util.DeleteUserWorkloadsConfigMap(env_resource, args.config_map_name, release_track=self.ReleaseTrack())
        log.status.Print('ConfigMap deleted')