from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from surface.run.services import update
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MultiRegionUpdate(update.Update):
    """Update environment variables and other configuration settings in Multi-Region Services."""

    @classmethod
    def Args(cls, parser):
        update.AlphaUpdate.Args(parser)
        flags.AddAddRegionsArg(parser)
        flags.AddRemoveRegionsArg(parser)

    def _ConnectionContext(self, args):
        return connection_context.GetConnectionContext(args, flags.Product.RUN, self.ReleaseTrack(), is_multiregion=True)

    def _GetBaseChanges(self, args, existing_service=None):
        changes = flags.GetServiceConfigurationChanges(args, base.ReleaseTrack) or []
        if flags.FlagIsExplicitlySet(args, 'add_regions') or flags.FlagIsExplicitlySet(args, 'remove_regions'):
            changes.append(config_changes.RegionsChangeAnnotationChange(to_add=args.add_regions, to_remove=args.remove_regions))
            super()._AssertChanges(changes, super().input_flags + ', `--add-regions`, `remove-regions`', ignore_empty=False)
            ch2 = super()._GetBaseChanges(args, existing_service, ignore_empty=True)
            return ch2 + changes

    def Run(self, args):
        if platforms.GetPlatform() != platforms.PLATFORM_MANAGED:
            raise c_exceptions.InvalidArgumentException('--platform', 'Multi-region Services are only supported on managed platform.')
        return super().Run(args)