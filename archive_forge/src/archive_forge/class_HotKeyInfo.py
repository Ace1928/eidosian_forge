from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HotKeyInfo(_messages.Message):
    """Information about a hot key.

  Fields:
    hotKeyAge: The age of the hot key measured from when it was first
      detected.
    key: A detected hot key that is causing limited parallelism. This field
      will be populated only if the following flag is set to true: "--
      enable_hot_key_logging".
    keyTruncated: If true, then the above key is truncated and cannot be
      deserialized. This occurs if the key above is populated and the key size
      is >5MB.
  """
    hotKeyAge = _messages.StringField(1)
    key = _messages.StringField(2)
    keyTruncated = _messages.BooleanField(3)