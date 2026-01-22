from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class KubeRun(base.Group):
    """Top level command to interact with KubeRun.

  Use this set of commands to create and manage KubeRun resources
  and some related project settings.
  """
    category = base.COMPUTE_CATEGORY
    detailed_help = {'EXAMPLES': '          To list your KubeRun services, run:\n\n            $ {command} core services list\n      '}

    def Filter(self, context, args):
        """Runs before any commands in this group."""
        base.RequireProjectID(args)
        del context, args