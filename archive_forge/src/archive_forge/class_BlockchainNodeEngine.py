from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class BlockchainNodeEngine(base.Group):
    """Create and manipulate Blockchain Node Engine resources."""
    detailed_help = DETAILED_HELP
    category = base.WEB3_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.EnableUserProjectQuota()