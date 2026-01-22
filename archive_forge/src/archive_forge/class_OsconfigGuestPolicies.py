from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class OsconfigGuestPolicies(base.Group):
    """Manage guest OS policies for Compute Engine VM instances."""