from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Gke(base.Group):
    """Create Dataproc GKE-based virtual clusters.

  All interactions other than creation should be handled by
  "gcloud dataproc clusters" commands.

  ## EXAMPLES

  To create a cluster, run:

    $ {command} my-cluster --region='us-central1' --gke-cluster='my-gke-cluster'
    --spark-engine-version='latest' --pools='name=dp,roles=default'
  """