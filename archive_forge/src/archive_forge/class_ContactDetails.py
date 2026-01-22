from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContactDetails(_messages.Message):
    """Details about specific contacts

  Fields:
    contacts: A list of contacts
  """
    contacts = _messages.MessageField('Contact', 1, repeated=True)