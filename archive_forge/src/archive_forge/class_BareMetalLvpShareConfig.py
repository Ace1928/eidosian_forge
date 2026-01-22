from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalLvpShareConfig(_messages.Message):
    """Specifies the configs for local persistent volumes under a shared file
  system.

  Fields:
    lvpConfig: Required. Defines the machine path and storage class for the
      LVP Share.
    sharedPathPvCount: The number of subdirectories to create under path.
  """
    lvpConfig = _messages.MessageField('BareMetalLvpConfig', 1)
    sharedPathPvCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)