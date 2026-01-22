from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsTriggersDeleteRequest(_messages.Message):
    """A EventarcProjectsLocationsTriggersDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the trigger is not found, the request
      will succeed but no action will be taken on the server.
    etag: If provided, the trigger will only be deleted if the etag matches
      the current etag on the resource.
    name: Required. The name of the trigger to be deleted.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)