from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Attestation(_messages.Message):
    """Occurrence that represents a single "attestation". The authenticity of
  an attestation can be verified using the attached signature. If the verifier
  trusts the public key of the signer, then verifying the signature is
  sufficient to establish trust. In this circumstance, the authority to which
  this attestation is attached is primarily useful for look-up (how to find
  this attestation if you already know the authority and artifact to be
  verified) and intent (which authority was this attestation intended to sign
  for).

  Fields:
    genericSignedAttestation: A GenericSignedAttestation attribute.
    pgpSignedAttestation: A PGP signed attestation.
  """
    genericSignedAttestation = _messages.MessageField('GenericSignedAttestation', 1)
    pgpSignedAttestation = _messages.MessageField('PgpSignedAttestation', 2)