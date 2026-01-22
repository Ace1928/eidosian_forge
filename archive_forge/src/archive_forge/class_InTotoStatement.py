from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InTotoStatement(_messages.Message):
    """Spec defined at https://github.com/in-
  toto/attestation/tree/main/spec#statement The serialized InTotoStatement
  will be stored as Envelope.payload. Envelope.payloadType is always
  "application/vnd.in-toto+json".

  Fields:
    _type: Always `https://in-toto.io/Statement/v0.1`.
    predicateType: `https://slsa.dev/provenance/v0.1` for SlsaProvenance.
    provenance: A InTotoProvenance attribute.
    slsaProvenance: A SlsaProvenance attribute.
    slsaProvenanceZeroTwo: A SlsaProvenanceZeroTwo attribute.
    subject: A Subject attribute.
  """
    _type = _messages.StringField(1)
    predicateType = _messages.StringField(2)
    provenance = _messages.MessageField('InTotoProvenance', 3)
    slsaProvenance = _messages.MessageField('SlsaProvenance', 4)
    slsaProvenanceZeroTwo = _messages.MessageField('SlsaProvenanceZeroTwo', 5)
    subject = _messages.MessageField('Subject', 6, repeated=True)