from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesStartWithEncryptionKeyRequest(_messages.Message):
    """A InstancesStartWithEncryptionKeyRequest object.

  Fields:
    disks: Array of disks associated with this instance that are protected
      with a customer-supplied encryption key. In order to start the instance,
      the disk url and its corresponding key must be provided. If the disk is
      not protected with a customer-supplied encryption key it should not be
      specified.
  """
    disks = _messages.MessageField('CustomerEncryptionKeyProtectedDisk', 1, repeated=True)