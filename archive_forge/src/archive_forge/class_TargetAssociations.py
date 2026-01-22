from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class TargetAssociations(base.Group):
    """Manage Authorization Toolkit TargetAssociations."""
    category = base.MANAGEMENT_TOOLS_CATEGORY