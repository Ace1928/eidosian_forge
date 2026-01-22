from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Workstations(base.Group):
    """Manage Cloud Workstations resources."""
    category = base.TOOLS_CATEGORY

    def Filter(self, context, args):
        del context, args