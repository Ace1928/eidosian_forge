from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Batches(base.Group):
    """Submit Dataproc batch jobs.

  Submit Dataproc batch jobs.

  Submit a job:

    $ {command} submit

  List all batch jobs:

    $ {command} list

  List job details:

    $ {command} describe JOB_ID

  Delete a batch job:

    $ {command} delete JOB_ID

  Cancel a running batch job without removing the batch resource:

    $ {command} cancel JOB_ID

  View job output:

    $ {command} wait JOB_ID
  """