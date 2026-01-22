from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebAuthn(_messages.Message):
    """Security key information specific to the Web Authentication protocol.

  Fields:
    rpId: Relying party ID for Web Authentication.
  """
    rpId = _messages.StringField(1)