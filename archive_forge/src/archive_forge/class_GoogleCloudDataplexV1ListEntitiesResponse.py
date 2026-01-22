from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListEntitiesResponse(_messages.Message):
    """List metadata entities response.

  Fields:
    entities: Entities in the specified parent zone.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no remaining results in the list.
  """
    entities = _messages.MessageField('GoogleCloudDataplexV1Entity', 1, repeated=True)
    nextPageToken = _messages.StringField(2)