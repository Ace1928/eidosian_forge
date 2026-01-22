from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DSSEAttestationNote(_messages.Message):
    """A DSSEAttestationNote object.

  Fields:
    hint: DSSEHint hints at the purpose of the attestation authority.
  """
    hint = _messages.MessageField('DSSEHint', 1)