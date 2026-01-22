from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntakeVlanAttachment(_messages.Message):
    """A GCP vlan attachment.

  Fields:
    id: Identifier of the VLAN attachment.
    pairingKey: Attachment pairing key.
  """
    id = _messages.StringField(1)
    pairingKey = _messages.StringField(2)