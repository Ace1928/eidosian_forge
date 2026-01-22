from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysKeysLookupKeyRequest(_messages.Message):
    """A ApikeysKeysLookupKeyRequest object.

  Fields:
    keyString: Required. Finds the project that owns the key string value.
  """
    keyString = _messages.StringField(1)