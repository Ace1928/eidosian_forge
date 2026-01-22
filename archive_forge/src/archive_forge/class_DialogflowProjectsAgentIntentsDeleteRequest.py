from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentIntentsDeleteRequest(_messages.Message):
    """A DialogflowProjectsAgentIntentsDeleteRequest object.

  Fields:
    name: Required. The name of the intent to delete. If this intent has
      direct or indirect followup intents, we also delete them. Format:
      `projects//agent/intents/`.
  """
    name = _messages.StringField(1, required=True)