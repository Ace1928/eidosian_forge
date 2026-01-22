from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListAspectTypesResponse(_messages.Message):
    """List AspectTypes response

  Fields:
    aspectTypes: ListAspectTypes under the given parent location.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachableLocations: Locations that could not be reached.
  """
    aspectTypes = _messages.MessageField('GoogleCloudDataplexV1AspectType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachableLocations = _messages.StringField(3, repeated=True)