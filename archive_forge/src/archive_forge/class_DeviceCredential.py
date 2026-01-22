from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeviceCredential(_messages.Message):
    """A server-stored device credential used for authentication.

  Fields:
    expirationTime: [Optional] The time at which this credential becomes
      invalid. This credential will be ignored for new client authentication
      requests after this timestamp; however, it will not be automatically
      deleted.
    publicKey: A public key used to verify the signature of JSON Web Tokens
      (JWTs). When adding a new device credential, either via device creation
      or via modifications, this public key credential may be required to be
      signed by one of the registry level certificates. More specifically, if
      the registry contains at least one certificate, any new device
      credential must be signed by one of the registry certificates. As a
      result, when the registry contains certificates, only X.509 certificates
      are accepted as device credentials. However, if the registry does not
      contain a certificate, self-signed certificates and public keys will be
      accepted. New device credentials must be different from every registry-
      level certificate.
  """
    expirationTime = _messages.StringField(1)
    publicKey = _messages.MessageField('PublicKeyCredential', 2)