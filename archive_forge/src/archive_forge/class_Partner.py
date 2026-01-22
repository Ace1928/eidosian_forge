from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Partner(base.Group):
    """Manage Secure Service Edge partner resources."""
    category = base.NETWORK_SECURITY_CATEGORY