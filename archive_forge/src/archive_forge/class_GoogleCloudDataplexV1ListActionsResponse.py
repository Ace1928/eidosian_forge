from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListActionsResponse(_messages.Message):
    """List actions response.

  Fields:
    actions: Actions under the given parent lake/zone/asset.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    actions = _messages.MessageField('GoogleCloudDataplexV1Action', 1, repeated=True)
    nextPageToken = _messages.StringField(2)