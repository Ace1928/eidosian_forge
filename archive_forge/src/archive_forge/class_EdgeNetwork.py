from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class EdgeNetwork(base.Group):
    """Manage Distributed Cloud Edge Network resources."""
    category = base.COMPUTE_CATEGORY

    def Filter(self, context, args):
        del context, args