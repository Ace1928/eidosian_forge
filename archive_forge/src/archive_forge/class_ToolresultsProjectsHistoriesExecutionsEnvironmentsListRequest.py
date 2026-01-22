from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsEnvironmentsListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsEnvironmentsListRequest object.

  Fields:
    executionId: Required. An Execution id.
    historyId: Required. A History id.
    pageSize: The maximum number of Environments to fetch. Default value: 25.
      The server will use this default if the field is not set or has a value
      of 0.
    pageToken: A continuation token to resume the query at the next item.
    projectId: Required. A Project id.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)