from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsTestCasesListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsTestCasesListRequest
  object.

  Fields:
    executionId: A Execution id Required.
    historyId: A History id. Required.
    pageSize: The maximum number of TestCases to fetch. Default value: 100.
      The server will use this default if the field is not set or has a value
      of 0. Optional.
    pageToken: A continuation token to resume the query at the next item.
      Optional.
    projectId: A Project id. Required.
    stepId: A Step id. Note: This step must include a TestExecutionStep.
      Required.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)
    stepId = _messages.StringField(6, required=True)