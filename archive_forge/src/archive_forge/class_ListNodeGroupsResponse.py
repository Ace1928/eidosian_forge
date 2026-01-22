from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNodeGroupsResponse(_messages.Message):
    """A response to a request to list the node groups in a cluster.

  Fields:
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    nodeGroups: The node groups in the cluster.
  """
    nextPageToken = _messages.StringField(1)
    nodeGroups = _messages.MessageField('NodeGroup', 2, repeated=True)