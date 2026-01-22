from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1ListDevicesResponse(_messages.Message):
    """Response message that is returned from the ListDevices method.

  Fields:
    devices: Devices meeting the list restrictions.
    nextPageToken: Token to retrieve the next page of results. Empty if there
      are no more results.
  """
    devices = _messages.MessageField('GoogleAppsCloudidentityDevicesV1Device', 1, repeated=True)
    nextPageToken = _messages.StringField(2)