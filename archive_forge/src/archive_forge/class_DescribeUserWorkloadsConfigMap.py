import textwrap
import typing
from typing import Union
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_config_maps_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DescribeUserWorkloadsConfigMap(base.Command):
    """Get details about a user workloads ConfigMap."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        base.Argument('config_map_name', nargs='?', help='Name of the ConfigMap.').AddToParser(parser)
        resource_args.AddEnvironmentResourceArg(parser, 'of the config_map', positional=False)

    def Run(self, args) -> Union['composer_v1alpha2_messages.UserWorkloadsConfigMap', 'composer_v1beta1_messages.UserWorkloadsConfigMap', 'composer_v1_messages.UserWorkloadsConfigMap']:
        env_resource = args.CONCEPTS.environment.Parse()
        return environments_user_workloads_config_maps_util.GetUserWorkloadsConfigMap(env_resource, args.config_map_name, release_track=self.ReleaseTrack())