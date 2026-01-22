from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleTypeTimeOfDay(_messages.Message):
    """Represents a time of day. The date and time zone are either not
  significant or are specified elsewhere. An API may choose to allow leap
  seconds. Related types are google.type.Date and `google.protobuf.Timestamp`.

  Fields:
    hours: Hours of day in 24 hour format. Should be from 0 to 23. An API may
      choose to allow the value "24:00:00" for scenarios like business closing
      time.
    minutes: Minutes of hour of day. Must be from 0 to 59.
    nanos: Fractions of seconds in nanoseconds. Must be from 0 to 999,999,999.
    seconds: Seconds of minutes of the time. Must normally be from 0 to 59. An
      API may allow the value 60 if it allows leap-seconds.
  """
    hours = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minutes = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    nanos = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    seconds = _messages.IntegerField(4, variant=_messages.Variant.INT32)