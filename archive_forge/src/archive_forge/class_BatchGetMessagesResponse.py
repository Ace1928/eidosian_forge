from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetMessagesResponse(_messages.Message):
    """Gets multiple messages in a specified HL7v2 store.

  Fields:
    messages: The returned Messages. See `MessageView` for populated fields.
  """
    messages = _messages.MessageField('Message', 1, repeated=True)