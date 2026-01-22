from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class PersistentResourcePreGA(base.Group):
    """Create and manage Vertex AI rersistent resources."""
    category = base.VERTEX_AI_CATEGORY