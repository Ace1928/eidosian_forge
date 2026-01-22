from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UseValueValuesEnum(_messages.Enum):
    """Required. The purpose of the key.

    Values:
      KEY_USE_UNSPECIFIED: The key use is not known.
      ENCRYPTION: The public key is used for encryption purposes.
    """
    KEY_USE_UNSPECIFIED = 0
    ENCRYPTION = 1