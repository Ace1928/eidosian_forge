from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsDeleteRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsDeleteRequest object.

  Fields:
    name: Required. The resource name of the `ClusterGroup` to be deleted.
    requestId: UUID of this invocation for idempotent operation.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)