from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkbenchQueryRequest(_messages.Message):
    """Message for querying a Workbench

  Fields:
    contents: Optional. The content of the current conversation with the
      model. For single-turn queries, this is a single instance. For multi-
      turn queries, this is a repeated field that contains conversation
      history + latest request.
    query: Required. The query from user.
    safetySettings: Optional. Per request settings for blocking unsafe
      content. Enforced on GenerateContentResponse.candidates.
  """
    contents = _messages.MessageField('Content', 1, repeated=True)
    query = _messages.StringField(2)
    safetySettings = _messages.MessageField('SafetySetting', 3, repeated=True)