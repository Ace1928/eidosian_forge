from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
@base.Hidden
class Overwatch(base.Group):
    """Manage Overwatch Commands."""
    category = base.SECURITY_CATEGORY