from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RemoteBuildExecution(base.Group):
    """Manage Remote Build Execution.

  Implementation for commands for Remote Build Execution Admin API integration.
  """
    category = base.CI_CD_CATEGORY

    def Filter(self, context, args):
        del context, args