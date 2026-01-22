from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScratchDisksValueListEntry(_messages.Message):
    """A ScratchDisksValueListEntry object.

    Fields:
      diskGb: Size of the scratch disk, defined in GB.
    """
    diskGb = _messages.IntegerField(1, variant=_messages.Variant.INT32)