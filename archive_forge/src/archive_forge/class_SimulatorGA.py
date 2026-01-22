from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SimulatorGA(base.Group):
    """Understand how an IAM policy change could impact access before deploying the change.
  """

    def Filter(self, context, args):
        """Enables User-Project override for this surface."""
        base.EnableUserProjectQuota()