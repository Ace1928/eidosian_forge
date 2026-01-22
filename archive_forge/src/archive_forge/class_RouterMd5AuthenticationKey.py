from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterMd5AuthenticationKey(_messages.Message):
    """A RouterMd5AuthenticationKey object.

  Fields:
    key: [Input only] Value of the key. For patch and update calls, it can be
      skipped to copy the value from the previous configuration. This is
      allowed if the key with the same name existed before the operation.
      Maximum length is 80 characters. Can only contain printable ASCII
      characters.
    name: Name used to identify the key. Must be unique within a router. Must
      be referenced by exactly one bgpPeer. Must comply with RFC1035.
  """
    key = _messages.StringField(1)
    name = _messages.StringField(2)