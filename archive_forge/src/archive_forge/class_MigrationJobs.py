from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class MigrationJobs(base.Group):
    """Manage Database Migration Service migration jobs.

  Commands for managing Database Migration Service migration jobs.
  """