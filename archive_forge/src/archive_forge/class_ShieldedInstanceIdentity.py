from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedInstanceIdentity(_messages.Message):
    """A Shielded Instance Identity.

  Fields:
    encryptionKey: An Endorsement Key (EK) made by the RSA 2048 algorithm
      issued to the Shielded Instance's vTPM.
    kind: [Output Only] Type of the resource. Always
      compute#shieldedInstanceIdentity for shielded Instance identity entry.
    signingKey: An Attestation Key (AK) made by the RSA 2048 algorithm issued
      to the Shielded Instance's vTPM.
  """
    encryptionKey = _messages.MessageField('ShieldedInstanceIdentityEntry', 1)
    kind = _messages.StringField(2, default='compute#shieldedInstanceIdentity')
    signingKey = _messages.MessageField('ShieldedInstanceIdentityEntry', 3)