from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class Manage(base.Group):
    """Manage Cloud SCC (Security Command Center) settings."""
    category = base.SECURITY_CATEGORY