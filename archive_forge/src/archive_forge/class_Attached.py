from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Attached(base.Group):
    """Manage Attached clusters for running containers."""
    category = base.COMPUTE_CATEGORY

    def Filter(self, context, args):
        del context, args