from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EffectiveGuestPolicySourcedPackageRepository(_messages.Message):
    """A guest policy package repository including its source.

  Fields:
    packageRepository: A software package repository to configure on the VM
      instance.
    source: Name of the guest policy providing this config.
  """
    packageRepository = _messages.MessageField('PackageRepository', 1)
    source = _messages.StringField(2)