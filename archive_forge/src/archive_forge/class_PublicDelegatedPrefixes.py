from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class PublicDelegatedPrefixes(base.Group):
    """Manage public delegated prefix resources."""
    category = base.NETWORKING_CATEGORY