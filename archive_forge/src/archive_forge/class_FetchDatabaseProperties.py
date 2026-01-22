from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
class FetchDatabaseProperties(base.Command):
    """Fetch database properties."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'for which to fetch database properties')

    def Run(self, args):
        env_ref = args.CONCEPTS.environment.Parse()
        release_track = self.ReleaseTrack()
        return environments_api_util.FetchDatabaseProperties(env_ref, release_track=release_track)