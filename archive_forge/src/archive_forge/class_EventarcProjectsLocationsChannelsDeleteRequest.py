from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsChannelsDeleteRequest(_messages.Message):
    """A EventarcProjectsLocationsChannelsDeleteRequest object.

  Fields:
    name: Required. The name of the channel to be deleted.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    name = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)