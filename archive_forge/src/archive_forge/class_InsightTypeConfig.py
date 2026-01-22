from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class InsightTypeConfig(base.Group):
    """Manage Cloud resource insight type configuration."""