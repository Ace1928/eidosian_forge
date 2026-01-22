from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsGetRequest object.

  Fields:
    executionId: A Execution id. Required.
    historyId: A History id. Required.
    projectId: A Project id. Required.
    stepId: A Step id. Required.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    stepId = _messages.StringField(4, required=True)