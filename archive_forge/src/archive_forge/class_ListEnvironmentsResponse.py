from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEnvironmentsResponse(_messages.Message):
    """Response message for EnvironmentService.ListEnvironments.

  Fields:
    environments: Environments. Always set.
    executionId: A Execution id Always set.
    historyId: A History id. Always set.
    nextPageToken: A continuation token to resume the query at the next item.
      Will only be set if there are more Environments to fetch.
    projectId: A Project id. Always set.
  """
    environments = _messages.MessageField('Environment', 1, repeated=True)
    executionId = _messages.StringField(2)
    historyId = _messages.StringField(3)
    nextPageToken = _messages.StringField(4)
    projectId = _messages.StringField(5)