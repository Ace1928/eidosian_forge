from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddNodesRequest(_messages.Message):
    """Request for adding nodes to the given cluster until the target count is
  reached.

  Fields:
    nodeCount: Required. Number of desired bare metal nodes in this cluster.
  """
    nodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)