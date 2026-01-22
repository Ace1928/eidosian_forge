from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class InstanceSettings(base.Group):
    """Describe and update Compute Engine instance settings."""