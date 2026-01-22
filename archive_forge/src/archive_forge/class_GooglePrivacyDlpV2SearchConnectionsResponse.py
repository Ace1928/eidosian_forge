from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SearchConnectionsResponse(_messages.Message):
    """Response message for SearchConnections.

  Fields:
    connections: List of connections that match the search query. Note that
      only a subset of the fields will be populated, and only "name" is
      guaranteed to be set. For full details of a Connection, call
      GetConnection with the name.
    nextPageToken: Token to retrieve the next page of results. An empty value
      means there are no more results.
  """
    connections = _messages.MessageField('GooglePrivacyDlpV2Connection', 1, repeated=True)
    nextPageToken = _messages.StringField(2)