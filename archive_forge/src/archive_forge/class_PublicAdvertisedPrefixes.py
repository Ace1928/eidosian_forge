from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class PublicAdvertisedPrefixes(base.Group):
    """Manage public advertised prefix resources."""
    category = base.NETWORKING_CATEGORY