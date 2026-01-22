from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def CreateCluster(dataproc, cluster_ref, cluster, is_async, timeout, enable_create_on_gke=False, action_on_failed_primary_workers=None):
    """Create a cluster.

  Args:
    dataproc: Dataproc object that contains client, messages, and resources
    cluster_ref: Full resource ref of cluster with name, region, and project id
    cluster: Cluster to create
    is_async: Whether to wait for the operation to complete
    timeout: Timeout used when waiting for the operation to complete
    enable_create_on_gke: Whether to enable creation of GKE-based clusters
    action_on_failed_primary_workers: Action to be performed when primary
      workers fail during cluster creation. Should be None for dataproc of
      v1beta2 version

  Returns:
    Created cluster, or None if async
  """
    request_id = util.GetUniqueId()
    request = dataproc.GetCreateClusterRequest(cluster=cluster, project_id=cluster_ref.projectId, region=cluster_ref.region, request_id=request_id, action_on_failed_primary_workers=action_on_failed_primary_workers)
    operation = dataproc.client.projects_regions_clusters.Create(request)
    if is_async:
        log.status.write('Creating [{0}] with operation [{1}].'.format(cluster_ref, operation.name))
        return
    operation = util.WaitForOperation(dataproc, operation, message='Waiting for cluster creation operation', timeout_s=timeout)
    get_request = dataproc.messages.DataprocProjectsRegionsClustersGetRequest(projectId=cluster_ref.projectId, region=cluster_ref.region, clusterName=cluster_ref.clusterName)
    cluster = dataproc.client.projects_regions_clusters.Get(get_request)
    if cluster.status.state == dataproc.messages.ClusterStatus.StateValueValuesEnum.RUNNING:
        if enable_create_on_gke and cluster.config.gkeClusterConfig is not None:
            log.CreatedResource(cluster_ref, details='Cluster created on GKE cluster {0}'.format(cluster.config.gkeClusterConfig.namespacedGkeDeploymentTarget.targetGkeCluster))
        elif cluster.virtualClusterConfig is not None:
            if cluster.virtualClusterConfig.kubernetesClusterConfig.gkeClusterConfig is not None:
                log.CreatedResource(cluster_ref, details='Virtual Cluster created on GKE cluster: {0}'.format(cluster.virtualClusterConfig.kubernetesClusterConfig.gkeClusterConfig.gkeClusterTarget))
        else:
            zone_uri = cluster.config.gceClusterConfig.zoneUri
            zone_short_name = zone_uri.split('/')[-1]
            log.CreatedResource(cluster_ref, details='Cluster placed in zone [{0}]'.format(zone_short_name))
    else:
        log.error('Create cluster failed!')
        if cluster.status.detail:
            log.error('Details:\n' + cluster.status.detail)
    return cluster