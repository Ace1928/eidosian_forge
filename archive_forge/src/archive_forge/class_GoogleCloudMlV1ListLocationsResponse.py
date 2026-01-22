from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListLocationsResponse(_messages.Message):
    """A GoogleCloudMlV1ListLocationsResponse object.

  Fields:
    locations: Locations where at least one type of CMLE capability is
      available.
    nextPageToken: Optional. Pass this token as the `page_token` field of the
      request for a subsequent call.
  """
    locations = _messages.MessageField('GoogleCloudMlV1Location', 1, repeated=True)
    nextPageToken = _messages.StringField(2)