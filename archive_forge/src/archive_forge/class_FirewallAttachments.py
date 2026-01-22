from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class FirewallAttachments(base.Group):
    """Create and manage Firewall attachments."""
    category = base.NETWORK_SECURITY_CATEGORY