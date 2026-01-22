from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContactInfo(_messages.Message):
    """Contact information of stakeholders.

  Fields:
    channel: Optional. Communication channel of the contacts.
    displayName: Optional. Contact's name. Can have a maximum length of 63
      characters.
    email: Required. Email address of the contacts.
  """
    channel = _messages.MessageField('Channel', 1)
    displayName = _messages.StringField(2)
    email = _messages.StringField(3)