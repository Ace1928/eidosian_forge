from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeBytes(_messages.Message):
    """Bytes Values of type `Bytes` are stored in `Value.bytes_value`.

  Fields:
    encoding: The encoding to use when converting to/from lower level types.
  """
    encoding = _messages.MessageField('GoogleBigtableAdminV2TypeBytesEncoding', 1)