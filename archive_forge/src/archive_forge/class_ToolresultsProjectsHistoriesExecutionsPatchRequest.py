from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsPatchRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsPatchRequest object.

  Fields:
    execution: A Execution resource to be passed as the request body.
    executionId: Required.
    historyId: Required.
    projectId: A Project id. Required.
    requestId: A unique request ID for server to detect duplicated requests.
      For example, a UUID. Optional, but strongly recommended.
  """
    execution = _messages.MessageField('Execution', 1)
    executionId = _messages.StringField(2, required=True)
    historyId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)