from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsLocationsClustersNodePoolsCompleteUpgradeRequest(_messages.Message):
    """A ContainerProjectsLocationsClustersNodePoolsCompleteUpgradeRequest
  object.

  Fields:
    completeNodePoolUpgradeRequest: A CompleteNodePoolUpgradeRequest resource
      to be passed as the request body.
    name: The name (project, location, cluster, node pool id) of the node pool
      to complete upgrade. Specified in the format
      `projects/*/locations/*/clusters/*/nodePools/*`.
  """
    completeNodePoolUpgradeRequest = _messages.MessageField('CompleteNodePoolUpgradeRequest', 1)
    name = _messages.StringField(2, required=True)