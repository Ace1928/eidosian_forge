from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class IpBlocks(base.Group):
    """Explore IP blocks used by Firebase Test Lab devices."""
    detailed_help = {'DESCRIPTION': '\n          Get a list of IP address blocks in CIDR notation used by Firebase Test\n          Lab devices.\n          ', 'EXAMPLES': '\n          To see the list of the IP blocks used, their form factors, and\n          the date they were added to Firebase Test Lab, run:\n\n            $ {command} list\n      '}

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags common to this sub-group.

    Args:
      parser: An argparse parser used to add arguments that immediately follow
          this group in the CLI. Positional arguments are allowed.
    """
        pass