from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class Aws(base.Group):
    """Deploy and manage clusters of machines on AWS for running containers."""
    category = base.COMPUTE_CATEGORY

    def Filter(self, context, args):
        del context, args