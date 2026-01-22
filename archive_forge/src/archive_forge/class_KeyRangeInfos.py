from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyRangeInfos(_messages.Message):
    """A message representing a list of specific information for multiple key
  ranges.

  Fields:
    infos: The list individual KeyRangeInfos.
    totalSize: The total size of the list of all KeyRangeInfos. This may be
      larger than the number of repeated messages above. If that is the case,
      this number may be used to determine how many are not being shown.
  """
    infos = _messages.MessageField('KeyRangeInfo', 1, repeated=True)
    totalSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)