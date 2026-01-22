from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentInfrastructureSpecComputeResources(_messages.Message):
    """Compute resources associated with the analyze interactive workloads.

  Fields:
    diskSizeGb: Optional. Size in GB of the disk. Default is 100 GB.
    maxNodeCount: Optional. Max configurable nodes. If max_node_count >
      node_count, then auto-scaling is enabled.
    nodeCount: Optional. Total number of nodes in the sessions created for
      this environment.
  """
    diskSizeGb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxNodeCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    nodeCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)