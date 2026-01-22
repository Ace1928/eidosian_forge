from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DomainMappings(base.Group):
    """View and manage your Cloud Run for Anthos domain mappings.

  This set of commands can be used to view and manage your service's domain
  mappings.

  To view and manage fully managed Cloud Run domain mappings, use
  `gcloud beta run domain-mappings`.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To list your Cloud Run domain mappings, run:\n\n            $ {command} list\n      '}

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser, anthos_only=True)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        self._CheckPlatform()
        return context

    def _CheckPlatform(self):
        if platforms.GetPlatform() == platforms.PLATFORM_MANAGED:
            raise exceptions.PlatformError('This command group is in beta for fully managed Cloud Run; use `gcloud beta run domain-mappings`.')