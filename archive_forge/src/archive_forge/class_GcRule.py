from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcRule(_messages.Message):
    """Rule for determining which cells to delete during garbage collection.

  Fields:
    intersection: Delete cells that would be deleted by every nested rule.
    maxAge: Delete cells in a column older than the given age. Values must be
      at least one millisecond, and will be truncated to microsecond
      granularity.
    maxNumVersions: Delete all cells in a column except the most recent N.
    union: Delete cells that would be deleted by any nested rule.
  """
    intersection = _messages.MessageField('Intersection', 1)
    maxAge = _messages.StringField(2)
    maxNumVersions = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    union = _messages.MessageField('Union', 4)