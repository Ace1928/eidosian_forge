from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1AttributeValue(_messages.Message):
    """The allowed types for [VALUE] in a `[KEY]:[VALUE]` attribute.

  Fields:
    boolValue: A Boolean value represented by `true` or `false`.
    intValue: A 64-bit signed integer.
    stringValue: A string up to 256 bytes long.
  """
    boolValue = _messages.BooleanField(1)
    intValue = _messages.IntegerField(2)
    stringValue = _messages.MessageField('GoogleApiServicecontrolV1TruncatableString', 3)