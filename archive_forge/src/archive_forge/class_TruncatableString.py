from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TruncatableString(_messages.Message):
    """Represents a string that might be shortened to a specified length.

  Fields:
    truncatedByteCount: The number of bytes removed from the original string.
      If this value is 0, then the string was not shortened.
    value: The shortened string. For example, if the original string is 500
      bytes long and the limit of the string is 128 bytes, then `value`
      contains the first 128 bytes of the 500-byte string. Truncation always
      happens on a UTF8 character boundary. If there are multi-byte characters
      in the string, then the length of the shortened string might be less
      than the size limit.
  """
    truncatedByteCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    value = _messages.StringField(2)