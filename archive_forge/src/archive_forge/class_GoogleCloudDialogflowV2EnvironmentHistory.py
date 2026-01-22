from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2EnvironmentHistory(_messages.Message):
    """The response message for Environments.GetEnvironmentHistory.

  Fields:
    entries: Output only. The list of agent environments. There will be a
      maximum number of items returned based on the page_size field in the
      request.
    nextPageToken: Output only. Token to retrieve the next page of results, or
      empty if there are no more results in the list.
    parent: Output only. The name of the environment this history is for.
      Supported formats: - `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
  """
    entries = _messages.MessageField('GoogleCloudDialogflowV2EnvironmentHistoryEntry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    parent = _messages.StringField(3)