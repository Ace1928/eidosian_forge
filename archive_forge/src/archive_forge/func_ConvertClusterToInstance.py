from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConvertClusterToInstance(cluster):
    """Convert a dataproc cluster to instance object.

  Args:
    cluster: cluster returned from Dataproc service.

  Returns:
    Instance: instance dict represents resources installed on GDCE cluster.
  """
    instance = dict()
    gdce_cluster_config = cluster.virtualClusterConfig.kubernetesClusterConfig.gdceClusterConfig
    instance['instanceName'] = cluster.clusterName
    instance['instanceUuid'] = cluster.clusterUuid
    instance['projectId'] = cluster.projectId
    instance['status'] = cluster.status
    instance['gdcEdgeIdentityProvider'] = gdce_cluster_config.gdcEdgeIdentityProvider
    instance['gdcEdgeMembershipTarget'] = gdce_cluster_config.gdcEdgeMembershipTarget
    instance['gdcEdgeWorkloadIdentityPool'] = gdce_cluster_config.gdcEdgeWorkloadIdentityPool
    return instance