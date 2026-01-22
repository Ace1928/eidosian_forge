from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListLakesResponse(_messages.Message):
    """List lakes response.

  Fields:
    lakes: Lakes under the given parent location.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachableLocations: Locations that could not be reached.
  """
    lakes = _messages.MessageField('GoogleCloudDataplexV1Lake', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachableLocations = _messages.StringField(3, repeated=True)