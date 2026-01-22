from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersNodePoolsDeleteRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersNodePoolsDeleteRequest object.

  Fields:
    name: Required. The resource name of the node pool.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)