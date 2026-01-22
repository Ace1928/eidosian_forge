from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotResourceUsageStat(_messages.Message):
    """A GoogleDevtoolsRemotebuildbotResourceUsageStat object.

  Fields:
    total: A string attribute.
    used: A string attribute.
  """
    total = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    used = _messages.IntegerField(2, variant=_messages.Variant.UINT64)