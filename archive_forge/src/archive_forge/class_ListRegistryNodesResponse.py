from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRegistryNodesResponse(_messages.Message):
    """Message for response to listing RegistryNodes

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    registryNodes: The list of RegistryNode
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    registryNodes = _messages.MessageField('RegistryNode', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)