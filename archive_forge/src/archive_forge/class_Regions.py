from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
class Regions(base.Group):
    """View available Cloud Run (fully managed) regions."""

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformArg(parser, managed_only=True)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        self._CheckPlatform()
        return context

    def _CheckPlatform(self):
        platform = platforms.GetPlatform()
        if platform is not None and platform != platforms.PLATFORM_MANAGED:
            raise exceptions.PlatformError('This command group only supports listing regions for Cloud Run (fully managed).')