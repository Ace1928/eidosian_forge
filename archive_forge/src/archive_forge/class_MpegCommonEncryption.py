from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MpegCommonEncryption(_messages.Message):
    """Configuration for MPEG Common Encryption (MPEG-CENC).

  Fields:
    scheme: Required. Specify the encryption scheme. Supported encryption
      schemes: - `cenc` - `cbcs`
  """
    scheme = _messages.StringField(1)