from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class JobTriggers(base.Group):
    """Cloud DLP commands for creating and managing Cloud DLP job triggers.

  Job triggers contain configurations to run Cloud DLP jobs on a set schedule.
  """