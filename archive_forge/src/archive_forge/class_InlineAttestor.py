from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InlineAttestor(_messages.Message):
    """An attestor that attests to container image artifacts. This attestor is
  to be inlined as part of the policy.

  Fields:
    attestationNote: Required. The Grafeas resource name of an ATTESTATION
      Note, created by the user, in the form of `projects/*/notes/*`. An
      attestation by this attestor is stored as a Grafeas Attestation
      Occurrence that names a container image and that links to this Note.
      Grafeas is an external dependency.
    description: Optional. A description, for information purposes only.
    id: Required. An id used to identify the attestor in the policy. It should
      be unique within the same policy.
    publicKeys: Optional. Public keys that verify attestations signed by this
      attestor. If this field is non-empty, one of the specified public keys
      must verify that an attestation was signed by this attestor for the
      image specified in the evaluation request. If this field is empty, this
      attestor always returns that no valid attestations exist.
  """
    attestationNote = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.StringField(3)
    publicKeys = _messages.MessageField('AttestorPublicKey', 4, repeated=True)