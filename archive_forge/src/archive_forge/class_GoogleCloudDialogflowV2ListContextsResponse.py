from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListContextsResponse(_messages.Message):
    """The response message for Contexts.ListContexts.

  Fields:
    contexts: The list of contexts. There will be a maximum number of items
      returned based on the page_size field in the request.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    contexts = _messages.MessageField('GoogleCloudDialogflowV2Context', 1, repeated=True)
    nextPageToken = _messages.StringField(2)