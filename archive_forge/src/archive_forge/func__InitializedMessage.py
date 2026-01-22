from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import gke_destinations
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _InitializedMessage():
    project = properties.VALUES.core.project.Get(required=True)
    trigger_cmd = 'gcloud eventarc triggers create'
    return 'Initialized project [{}] for Cloud Run for Anthos/GKE destinations in Eventarc. Next, create a trigger via `{}`.'.format(project, trigger_cmd)