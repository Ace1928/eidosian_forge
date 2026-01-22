from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddOperationEdge(_messages.Message):
    """A directed edge representing dependency between two RDDs.

  Fields:
    fromId: A integer attribute.
    toId: A integer attribute.
  """
    fromId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    toId = _messages.IntegerField(2, variant=_messages.Variant.INT32)