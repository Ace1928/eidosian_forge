from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WrappingPublicKey(_messages.Message):
    """The public key component of the wrapping key. For details of the type of
  key this public key corresponds to, see the ImportMethod.

  Fields:
    pem: The public key, encoded in PEM format. For more information, see the
      [RFC 7468](https://tools.ietf.org/html/rfc7468) sections for [General
      Considerations](https://tools.ietf.org/html/rfc7468#section-2) and
      [Textual Encoding of Subject Public Key Info]
      (https://tools.ietf.org/html/rfc7468#section-13).
  """
    pem = _messages.StringField(1)