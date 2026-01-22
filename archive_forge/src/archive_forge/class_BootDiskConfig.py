from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BootDiskConfig(_messages.Message):
    """Boot disk configurations.

  Fields:
    customerEncryptionKey: Optional. Customer encryption key for boot disk.
    enableConfidentialCompute: Optional. Whether the boot disk will be created
      with confidential compute mode.
  """
    customerEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 1)
    enableConfidentialCompute = _messages.BooleanField(2)