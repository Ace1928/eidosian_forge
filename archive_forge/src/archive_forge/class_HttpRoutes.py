from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class HttpRoutes(base.Group):
    """Manage Network Services HttpRoutes."""
    category = base.MANAGEMENT_TOOLS_CATEGORY