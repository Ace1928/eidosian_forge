from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from surface.run.services import describe
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MultiRegionDescribe(describe.Describe):
    """Describes multi-region service."""

    def _ConnectionContext(self, args):
        return connection_context.GetConnectionContext(args, flags.Product.RUN, self.ReleaseTrack(), is_multiregion=True)

    def Run(self, args):
        if platforms.GetPlatform() != platforms.PLATFORM_MANAGED:
            raise c_exceptions.InvalidArgumentException('--platform', 'Multi-region Services are only supported on managed platform.')
        return super().Run(args)