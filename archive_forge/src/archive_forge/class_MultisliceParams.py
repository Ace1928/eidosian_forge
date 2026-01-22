from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultisliceParams(_messages.Message):
    """Parameters to specify for multislice QueuedResource requests. This
  message must be populated in case of multislice requests instead of node_id.

  Fields:
    nodeCount: Required. Number of nodes with this spec. The system will
      attempt to provison "node_count" nodes as part of the request. This
      needs to be > 1.
    nodeIdPrefix: Optional. Prefix of node_ids in case of multislice request.
      Should follow the `^[A-Za-z0-9_.~+%-]+$` regex format. If node_count = 3
      and node_id_prefix = "np", node ids of nodes created will be "np-0",
      "np-1", "np-2". If this field is not provided we use queued_resource_id
      as the node_id_prefix.
  """
    nodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    nodeIdPrefix = _messages.StringField(2)