from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeleteBlockchainValidatorConfigResponse(_messages.Message):
    """Response message for DeleteBlockchainValidatorConfig

  Fields:
    slashingProtectionData: Optional. Slashing protection data for a set of
      Ethereum voting keys, as described in EIP-3076, deserialized into a
      string.
  """
    slashingProtectionData = _messages.StringField(1)