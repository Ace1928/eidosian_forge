from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
def PartialUpdate(cluster_ref, nodes=None, autoscaling_min=None, autoscaling_max=None, autoscaling_cpu_target=None, autoscaling_storage_target=None, disable_autoscaling=False):
    """Partially update a cluster.

  Args:
    cluster_ref: A resource reference to the cluster to update.
    nodes: int, the number of nodes in this cluster.
    autoscaling_min: int, the minimum number of nodes for autoscaling.
    autoscaling_max: int, the maximum number of nodes for autoscaling.
    autoscaling_cpu_target: int, the target CPU utilization percent for
      autoscaling.
    autoscaling_storage_target: int, the target storage utilization gibibytes
      per node for autoscaling.
    disable_autoscaling: bool, True means disable autoscaling if it is currently
      enabled. False means change nothing whether it is currently enabled or
      not.

  Returns:
    Long running operation.
  """
    client = util.GetAdminClient()
    msgs = util.GetAdminMessages()
    if disable_autoscaling:
        if autoscaling_min is not None or autoscaling_max is not None or autoscaling_cpu_target is not None or (autoscaling_storage_target is not None):
            raise ValueError('autoscaling arguments cannot be set together with disable_autoscaling')
        return client.projects_instances_clusters.PartialUpdateCluster(BuildPartialUpdateClusterRequest(msgs=msgs, name=cluster_ref.RelativeName(), nodes=nodes, update_mask='serve_nodes,cluster_config.cluster_autoscaling_config'))
    changed_fields = []
    if nodes is not None:
        changed_fields.append('serve_nodes')
    if autoscaling_min is not None:
        changed_fields.append('cluster_config.cluster_autoscaling_config.autoscaling_limits.min_serve_nodes')
    if autoscaling_max is not None:
        changed_fields.append('cluster_config.cluster_autoscaling_config.autoscaling_limits.max_serve_nodes')
    if autoscaling_cpu_target is not None:
        changed_fields.append('cluster_config.cluster_autoscaling_config.autoscaling_targets.cpu_utilization_percent')
    if autoscaling_storage_target is not None:
        changed_fields.append('cluster_config.cluster_autoscaling_config.autoscaling_targets.storage_utilization_gib_per_node')
    update_mask = ','.join(changed_fields)
    return client.projects_instances_clusters.PartialUpdateCluster(BuildPartialUpdateClusterRequest(msgs=msgs, name=cluster_ref.RelativeName(), nodes=nodes, autoscaling_min=autoscaling_min, autoscaling_max=autoscaling_max, autoscaling_cpu_target=autoscaling_cpu_target, autoscaling_storage_target=autoscaling_storage_target, update_mask=update_mask))