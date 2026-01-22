from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeInt64Encoding(_messages.Message):
    """Rules used to convert to/from lower level types.

  Fields:
    bigEndianBytes: Use `BigEndianBytes` encoding.
  """
    bigEndianBytes = _messages.MessageField('GoogleBigtableAdminV2TypeInt64EncodingBigEndianBytes', 1)