from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RangeValue(_messages.Message):
    """[Experimental] Defines the ranges for range partitioning.

    Fields:
      end: [Experimental] The end of range partitioning, exclusive.
      interval: [Experimental] The width of each interval.
      start: [Experimental] The start of range partitioning, inclusive.
    """
    end = _messages.IntegerField(1)
    interval = _messages.IntegerField(2)
    start = _messages.IntegerField(3)