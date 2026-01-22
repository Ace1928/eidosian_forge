from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.krmapihosting import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def GetGKECluster(name, location):
    """Fetches the information about the GKE cluster backing the Config Controller."""
    cluster_id = 'krmapihost-' + name
    location_id = location
    project = None
    gke_api = container_api_adapter.NewAPIAdapter('v1')
    log.status.Print('Fetching cluster endpoint and auth data.')
    cluster_ref = gke_api.ParseCluster(cluster_id, location_id, project)
    cluster = gke_api.GetCluster(cluster_ref)
    if not gke_api.IsRunning(cluster):
        log.warning(NOT_RUNNING_MSG.format(cluster_ref.clusterId))
    return (cluster, cluster_ref)