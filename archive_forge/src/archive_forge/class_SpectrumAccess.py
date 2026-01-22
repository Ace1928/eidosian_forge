from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class SpectrumAccess(base.Group):
    """Create and manage Spectrum Access System (SAS) resources."""
    category = base.MANAGEMENT_TOOLS_CATEGORY

    def Filter(self, context, args):
        del context, args