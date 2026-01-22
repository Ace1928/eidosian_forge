from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AttestorPublicKey(_messages.Message):
    """An attestor public key that will be used to verify attestations signed
  by this attestor.

  Fields:
    asciiArmoredPgpPublicKey: ASCII-armored representation of a PGP public
      key, as the entire output by the command `gpg --export --armor
      foo@example.com` (either LF or CRLF line endings). When using this
      field, `id` should be left blank. The BinAuthz API handlers will
      calculate the ID and fill it in automatically. BinAuthz computes this ID
      as the OpenPGP RFC4880 V4 fingerprint, represented as upper-case hex. If
      `id` is provided by the caller, it will be overwritten by the API-
      calculated ID.
    comment: Optional. A descriptive comment. This field may be updated.
    id: The ID of this public key. Signatures verified by BinAuthz must
      include the ID of the public key that can be used to verify them, and
      that ID must match the contents of this field exactly. Additional
      restrictions on this field can be imposed based on which public key type
      is encapsulated. See the documentation on `public_key` cases below for
      details.
    pkixPublicKey: A raw PKIX SubjectPublicKeyInfo format public key. NOTE:
      `id` may be explicitly provided by the caller when using this type of
      public key, but it MUST be a valid RFC3986 URI. If `id` is left blank, a
      default one will be computed based on the digest of the DER encoding of
      the public key.
  """
    asciiArmoredPgpPublicKey = _messages.StringField(1)
    comment = _messages.StringField(2)
    id = _messages.StringField(3)
    pkixPublicKey = _messages.MessageField('PkixPublicKey', 4)