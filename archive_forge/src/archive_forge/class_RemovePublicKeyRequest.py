from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemovePublicKeyRequest(_messages.Message):
    """Request message for RemovePublicKey.

  Fields:
    key: Key that should be removed from the environment.
  """
    key = _messages.StringField(1)