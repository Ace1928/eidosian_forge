from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudLocationListLocationsResponse(_messages.Message):
    """The response message for Locations.ListLocations.

  Fields:
    locations: A list of locations that matches the specified filter in the
      request.
    nextPageToken: The standard List next-page token.
  """
    locations = _messages.MessageField('GoogleCloudLocationLocation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)