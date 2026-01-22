from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class GKEBackup(base.Group):
    """Backup for GKE Services."""
    category = base.COMPUTE_CATEGORY

    def Filter(self, context, args):
        """See base class."""
        base.RequireProjectID(args)
        return context