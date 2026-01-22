from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeNodePoolConfig(_messages.Message):
    """The configuration of a GKE node pool used by a Dataproc-on-GKE cluster
  (https://cloud.google.com/dataproc/docs/concepts/jobs/dataproc-gke#create-a-
  dataproc-on-gke-cluster).

  Fields:
    autoscaling: Optional. The autoscaler configuration for this node pool.
      The autoscaler is enabled only when a valid configuration is present.
    config: Optional. The node pool configuration.
    locations: Optional. The list of Compute Engine zones
      (https://cloud.google.com/compute/docs/zones#available) where node pool
      nodes associated with a Dataproc on GKE virtual cluster will be
      located.Note: All node pools associated with a virtual cluster must be
      located in the same region as the virtual cluster, and they must be
      located in the same zone within that region.If a location is not
      specified during node pool creation, Dataproc on GKE will choose the
      zone.
  """
    autoscaling = _messages.MessageField('GkeNodePoolAutoscalingConfig', 1)
    config = _messages.MessageField('GkeNodeConfig', 2)
    locations = _messages.StringField(3, repeated=True)