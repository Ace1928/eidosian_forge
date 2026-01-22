from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import runtimes as runtime_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.notebooks import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Switch(base.Command):
    """Request for switching runtimes."""

    @classmethod
    def Args(cls, parser):
        """Register flags for this command."""
        api_version = util.ApiVersionSelector(cls.ReleaseTrack())
        flags.AddSwitchRuntimeFlags(api_version, parser)

    def Run(self, args):
        release_track = self.ReleaseTrack()
        client = util.GetClient(release_track)
        messages = util.GetMessages(release_track)
        runtime_service = client.projects_locations_runtimes
        operation = runtime_service.Switch(runtime_util.CreateRuntimeSwitchRequest(args, messages))
        return runtime_util.HandleLRO(operation, args, runtime_service, release_track, operation_type=runtime_util.OperationType.UPDATE)