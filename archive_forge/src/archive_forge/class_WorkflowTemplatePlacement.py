from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowTemplatePlacement(_messages.Message):
    """Specifies workflow execution target.Either managed_cluster or
  cluster_selector is required.

  Fields:
    clusterSelector: Optional. A selector that chooses target cluster for jobs
      based on metadata.The selector is evaluated at the time each job is
      submitted.
    managedCluster: A cluster that is managed by the workflow.
  """
    clusterSelector = _messages.MessageField('ClusterSelector', 1)
    managedCluster = _messages.MessageField('ManagedCluster', 2)