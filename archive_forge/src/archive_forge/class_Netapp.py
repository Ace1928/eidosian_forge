from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Netapp(base.Group):
    """Create and manipulate Cloud NetApp Files resources."""
    detailed_help = DETAILED_HELP
    category = base.STORAGE_CATEGORY