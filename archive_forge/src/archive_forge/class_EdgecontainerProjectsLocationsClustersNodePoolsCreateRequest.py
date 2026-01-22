from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersNodePoolsCreateRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersNodePoolsCreateRequest object.

  Fields:
    nodePool: A NodePool resource to be passed as the request body.
    nodePoolId: Required. A client-specified unique identifier for the node
      pool.
    parent: Required. The parent cluster where this node pool will be created.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
  """
    nodePool = _messages.MessageField('NodePool', 1)
    nodePoolId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)