from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class ServiceExtensions(base.Group):
    """Manage Service Extensions resources."""
    category = base.NETWORKING_CATEGORY

    def Filter(self, context, args):
        del context, args