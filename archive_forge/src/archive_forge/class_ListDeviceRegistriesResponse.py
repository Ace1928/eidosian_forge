from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDeviceRegistriesResponse(_messages.Message):
    """Response for `ListDeviceRegistries`.

  Fields:
    deviceRegistries: The registries that matched the query.
    nextPageToken: If not empty, indicates that there may be more registries
      that match the request; this value should be passed in a new
      `ListDeviceRegistriesRequest`.
  """
    deviceRegistries = _messages.MessageField('DeviceRegistry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)