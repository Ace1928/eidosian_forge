from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListSavedQueriesResponse(_messages.Message):
    """Response message for DatasetService.ListSavedQueries.

  Fields:
    nextPageToken: The standard List next-page token.
    savedQueries: A list of SavedQueries that match the specified filter in
      the request.
  """
    nextPageToken = _messages.StringField(1)
    savedQueries = _messages.MessageField('GoogleCloudAiplatformV1SavedQuery', 2, repeated=True)