from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceContent(_messages.Message):
    """The content of a single message from a participant.

  Fields:
    parts: The parts of the message.
    role: The role of the current conversation participant.
  """
    parts = _messages.MessageField('CloudAiNlLlmProtoServicePart', 1, repeated=True)
    role = _messages.StringField(2)