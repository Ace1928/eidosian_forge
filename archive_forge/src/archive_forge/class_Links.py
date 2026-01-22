from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Links(base.Group):
    """Manage linked datasets.

  Commands for managing linked datasets. A linked BigQuery dataset contains log
  data for the linked dataset's parent log bucket.
  """