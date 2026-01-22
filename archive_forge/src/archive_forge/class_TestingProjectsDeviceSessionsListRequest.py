from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsDeviceSessionsListRequest(_messages.Message):
    """A TestingProjectsDeviceSessionsListRequest object.

  Fields:
    filter: Optional. If specified, responses will be filtered by the given
      filter. Allowed fields are: session_state.
    pageSize: Optional. The maximum number of DeviceSessions to return.
    pageToken: Optional. A continuation token for paging.
    parent: Required. The name of the parent to request, e.g.
      "projects/{project_id}"
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)