from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class BareMetal(base.Group):
    """Deploy and manage Anthos clusters on bare metal for running containers."""
    category = base.COMPUTE_CATEGORY