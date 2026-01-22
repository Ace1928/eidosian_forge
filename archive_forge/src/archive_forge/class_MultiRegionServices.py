from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MultiRegionServices(base.Group):
    """Manage your Cloud Run resources."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser)

    def Filter(self, context, args):
        """Runs before any commands in this group."""
        base.RequireProjectID(args)
        del context, args