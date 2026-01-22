from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class DomainMappingsBeta(base.Group):
    """View and manage your Cloud Run domain mappings.

  This set of commands can be used to view and manage your service's domain
  mappings.
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To list your Cloud Run domain mappings, run:\n\n            $ {command} list\n      '}

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser)

    def _CheckPlatform(self):
        pass