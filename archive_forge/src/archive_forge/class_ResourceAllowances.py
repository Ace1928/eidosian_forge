from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class ResourceAllowances(base.Group):
    """Manage Batch resource allowance resources."""
    detailed_help = DETAILED_HELP
    category = base.BATCH_CATEGORY