from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ModelMonitoringJobs(base.Group):
    """Manage Vertex AI model monitoring jobs."""
    category = base.VERTEX_AI_CATEGORY