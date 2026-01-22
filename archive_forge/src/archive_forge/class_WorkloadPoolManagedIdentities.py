from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class WorkloadPoolManagedIdentities(base.Group):
    """Manage IAM workload identity pool managed identities.

  Commands for managing IAM workload identity pool managed identities.
  """