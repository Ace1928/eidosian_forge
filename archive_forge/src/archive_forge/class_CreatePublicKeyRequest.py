from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreatePublicKeyRequest(_messages.Message):
    """Request message for CreatePublicKey.

  Fields:
    key: Key that should be added to the environment.
  """
    key = _messages.MessageField('PublicKey', 1)