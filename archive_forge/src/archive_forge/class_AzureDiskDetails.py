from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureDiskDetails(_messages.Message):
    """The details of an Azure VM disk.

  Fields:
    diskId: Azure disk ID.
    diskNumber: The ordinal number of the disk.
    sizeGb: Size in GB.
  """
    diskId = _messages.StringField(1)
    diskNumber = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    sizeGb = _messages.IntegerField(3)