from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ActiveDirectory(base.Group):
    """Manage Managed Microsoft AD resources."""
    category = base.IDENTITY_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args