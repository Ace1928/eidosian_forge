from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class LbTrafficExtensions(base.Group):
    """Manage Service Extensions `LbTrafficExtension` resources."""