from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SecureServiceEdge(base.Group):
    """Manage Secure Service Edge resources."""
    category = base.NETWORK_SECURITY_CATEGORY