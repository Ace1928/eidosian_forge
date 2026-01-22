from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaEncryptionInfo(_messages.Message):
    """Message contains the JWT encryption information for the proxy server.

  Fields:
    encryptionSaEmail: Optional. Service Account for encryption key.
    jwk: Optional. JWK in string.
  """
    encryptionSaEmail = _messages.StringField(1)
    jwk = _messages.StringField(2)