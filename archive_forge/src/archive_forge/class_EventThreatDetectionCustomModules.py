from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA)
class EventThreatDetectionCustomModules(base.Group):
    """Manage custom modules."""
    category = base.SECURITY_CATEGORY