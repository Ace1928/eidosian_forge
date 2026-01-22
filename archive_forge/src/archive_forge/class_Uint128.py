from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Uint128(_messages.Message):
    """A Uint128 object.

  Fields:
    high: A string attribute.
    low: A string attribute.
  """
    high = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    low = _messages.IntegerField(2, variant=_messages.Variant.UINT64)