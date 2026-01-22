from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Osconfig(base.Group):
    """Manage OS Config tasks for Compute Engine VM instances."""
    category = base.TOOLS_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuota()