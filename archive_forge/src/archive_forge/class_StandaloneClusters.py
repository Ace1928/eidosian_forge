from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.projects import util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class StandaloneClusters(base.Group):
    """Create and manage standalone clusters in Anthos on bare metal."""