from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMirroringEndpointGroupAssociationsResponse(_messages.Message):
    """Message for response to listing MirroringEndpointGroupAssociations

  Fields:
    mirroringEndpointGroupAssociations: The list of
      MirroringEndpointGroupAssociation
    nextPageToken: A token identifying a page of results the server should
      return.
  """
    mirroringEndpointGroupAssociations = _messages.MessageField('MirroringEndpointGroupAssociation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)