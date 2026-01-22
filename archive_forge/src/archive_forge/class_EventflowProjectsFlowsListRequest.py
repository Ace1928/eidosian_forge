from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EventflowProjectsFlowsListRequest(_messages.Message):
    """A EventflowProjectsFlowsListRequest object.

  Fields:
    pageSize: Requested page size. Server may return more items than
      requested. If unspecified, server will pick an appropriate default.
    pageToken: A token identifying a page of results the server should return.
    parent: The parent, of form "projects/{projectId}", from which the flows
      should be listed.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)