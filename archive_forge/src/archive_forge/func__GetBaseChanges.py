from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from surface.run.services import update
def _GetBaseChanges(self, args, existing_service=None):
    changes = flags.GetServiceConfigurationChanges(args, base.ReleaseTrack) or []
    if flags.FlagIsExplicitlySet(args, 'add_regions') or flags.FlagIsExplicitlySet(args, 'remove_regions'):
        changes.append(config_changes.RegionsChangeAnnotationChange(to_add=args.add_regions, to_remove=args.remove_regions))
        super()._AssertChanges(changes, super().input_flags + ', `--add-regions`, `remove-regions`', ignore_empty=False)
        ch2 = super()._GetBaseChanges(args, existing_service, ignore_empty=True)
        return ch2 + changes