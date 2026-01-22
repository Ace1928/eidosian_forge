from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class PersistentResourceGA(base.Group):
    """Create and manage Vertex AI persistent resources."""
    category = base.VERTEX_AI_CATEGORY