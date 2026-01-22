from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import environments_workloads_util as workloads_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListWorkloads(base.Command):
    """List Composer workloads, supported in Composer 3 environments or greater."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'for which to display workloads')

    def Run(self, args):
        env_ref = args.CONCEPTS.environment.Parse()
        release_track = self.ReleaseTrack()
        env_obj = environments_api_util.Get(env_ref, release_track=self.ReleaseTrack())
        if not image_versions_command_util.IsVersionComposer3Compatible(image_version=env_obj.config.softwareConfig.imageVersion):
            raise command_util.InvalidUserInputError(COMPOSER3_IS_REQUIRED_MSG.format(composer_version=flags.MIN_COMPOSER3_VERSION))
        workloads_service = workloads_util.EnvironmentsWorkloadsService(release_track)
        return workloads_service.List(env_ref)