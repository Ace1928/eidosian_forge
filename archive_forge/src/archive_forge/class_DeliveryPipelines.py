from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DeliveryPipelines(base.Group):
    """Create and manage Delivery Pipeline resources for Cloud Deploy."""
    category = base.CI_CD_CATEGORY