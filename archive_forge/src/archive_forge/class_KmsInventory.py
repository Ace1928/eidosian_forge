from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class KmsInventory(base.Group):
    """Manages the KMS Inventory and Key Tracking commands."""
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuota()