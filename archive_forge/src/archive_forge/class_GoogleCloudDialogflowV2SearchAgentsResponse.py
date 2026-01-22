from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SearchAgentsResponse(_messages.Message):
    """The response message for Agents.SearchAgents.

  Fields:
    agents: The list of agents. There will be a maximum number of items
      returned based on the page_size field in the request.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    agents = _messages.MessageField('GoogleCloudDialogflowV2Agent', 1, repeated=True)
    nextPageToken = _messages.StringField(2)