from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesResumeRequest(_messages.Message):
    """A InstancesResumeRequest object.

  Fields:
    disks: Array of disks associated with this instance that are protected
      with a customer-supplied encryption key. In order to resume the
      instance, the disk url and its corresponding key must be provided. If
      the disk is not protected with a customer-supplied encryption key it
      should not be specified.
    instanceEncryptionKey: Decrypts data associated with an instance that is
      protected with a customer-supplied encryption key. If the instance you
      are starting is protected with a customer-supplied encryption key, the
      correct key must be provided otherwise the instance resume will not
      succeed.
  """
    disks = _messages.MessageField('CustomerEncryptionKeyProtectedDisk', 1, repeated=True)
    instanceEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 2)