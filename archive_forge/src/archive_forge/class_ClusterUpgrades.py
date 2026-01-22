from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ClusterUpgrades(base.Group):
    """Configure the Fleet clusterupgrade feature.

  This fleet feature is used to configure fleet-based rollout sequencing.
  """
    category = base.COMPUTE_CATEGORY