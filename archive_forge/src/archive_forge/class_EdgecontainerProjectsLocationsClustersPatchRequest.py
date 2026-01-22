from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersPatchRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersPatchRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    name: Required. The resource name of the cluster.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
    updateMask: Field mask is used to specify the fields to be overwritten in
      the Cluster resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then all fields will be overwritten.
  """
    cluster = _messages.MessageField('Cluster', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)