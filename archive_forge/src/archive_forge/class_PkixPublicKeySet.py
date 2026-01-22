from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PkixPublicKeySet(_messages.Message):
    """A bundle of PKIX public keys, used to authenticate attestation
  signatures. Generally, a signature is considered to be authenticated by a
  `PkixPublicKeySet` if any of the public keys verify it (i.e. it is an "OR"
  of the keys).

  Fields:
    pkixPublicKeys: Required. `pkix_public_keys` must have at least one entry.
  """
    pkixPublicKeys = _messages.MessageField('PkixPublicKey', 1, repeated=True)