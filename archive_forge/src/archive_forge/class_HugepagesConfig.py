from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HugepagesConfig(_messages.Message):
    """Hugepages amount in both 2m and 1g size

  Fields:
    hugepageSize1g: Optional. Amount of 1G hugepages
    hugepageSize2m: Optional. Amount of 2M hugepages
  """
    hugepageSize1g = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    hugepageSize2m = _messages.IntegerField(2, variant=_messages.Variant.INT32)