from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsEnvironmentsGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsEnvironmentsGetRequest object.

  Fields:
    environmentId: Required. An Environment id.
    executionId: Required. An Execution id.
    historyId: Required. A History id.
    projectId: Required. A Project id.
  """
    environmentId = _messages.StringField(1, required=True)
    executionId = _messages.StringField(2, required=True)
    historyId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)