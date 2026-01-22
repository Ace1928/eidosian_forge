from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNodeTypesResponse(_messages.Message):
    """Response message for VmwareEngine.ListNodeTypes

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    nodeTypes: A list of Node Types.
    unreachable: Locations that could not be reached when making an aggregated
      query using wildcards.
  """
    nextPageToken = _messages.StringField(1)
    nodeTypes = _messages.MessageField('NodeType', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)