from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlsaCheck(_messages.Message):
    """A SLSA provenance attestation check, which ensures that images are built
  by a trusted builder using source code from its trusted repositories only.

  Fields:
    rules: Specifies a list of verification rules for the SLSA attestations.
      An image is considered compliant with the SlsaCheck if any of the rules
      are satisfied.
  """
    rules = _messages.MessageField('VerificationRule', 1, repeated=True)