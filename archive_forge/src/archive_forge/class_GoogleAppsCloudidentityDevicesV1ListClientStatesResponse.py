from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1ListClientStatesResponse(_messages.Message):
    """Response message that is returned in ListClientStates.

  Fields:
    clientStates: Client states meeting the list restrictions.
    nextPageToken: Token to retrieve the next page of results. Empty if there
      are no more results.
  """
    clientStates = _messages.MessageField('GoogleAppsCloudidentityDevicesV1ClientState', 1, repeated=True)
    nextPageToken = _messages.StringField(2)