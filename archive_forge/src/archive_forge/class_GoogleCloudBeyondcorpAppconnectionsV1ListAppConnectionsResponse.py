from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectionsV1ListAppConnectionsResponse(_messages.Message):
    """Response message for BeyondCorp.ListAppConnections.

  Fields:
    appConnections: A list of BeyondCorp AppConnections in the project.
    nextPageToken: A token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: A list of locations that could not be reached.
  """
    appConnections = _messages.MessageField('GoogleCloudBeyondcorpAppconnectionsV1AppConnection', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)