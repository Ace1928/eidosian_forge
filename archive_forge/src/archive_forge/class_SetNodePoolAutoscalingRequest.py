from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetNodePoolAutoscalingRequest(_messages.Message):
    """SetNodePoolAutoscalingRequest sets the autoscaler settings of a node
  pool.

  Fields:
    autoscaling: Required. Autoscaling configuration for the node pool.
    clusterId: Deprecated. The name of the cluster to upgrade. This field has
      been deprecated and replaced by the name field.
    name: The name (project, location, cluster, node pool) of the node pool to
      set autoscaler settings. Specified in the format
      `projects/*/locations/*/clusters/*/nodePools/*`.
    nodePoolId: Deprecated. The name of the node pool to upgrade. This field
      has been deprecated and replaced by the name field.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    autoscaling = _messages.MessageField('NodePoolAutoscaling', 1)
    clusterId = _messages.StringField(2)
    name = _messages.StringField(3)
    nodePoolId = _messages.StringField(4)
    projectId = _messages.StringField(5)
    zone = _messages.StringField(6)