from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketIntervals(_messages.Message):
    """Next free: 7

  Enums:
    DurationValueValuesEnum: From how long ago in the past these intervals
      were observed.
    TypeValueValuesEnum: The type of packets for which inter-packet intervals
      were computed.

  Fields:
    avgMs: Average observed inter-packet interval in milliseconds.
    duration: From how long ago in the past these intervals were observed.
    maxMs: Maximum observed inter-packet interval in milliseconds.
    minMs: Minimum observed inter-packet interval in milliseconds.
    numIntervals: Number of inter-packet intervals from which these statistics
      were derived.
    type: The type of packets for which inter-packet intervals were computed.
  """

    class DurationValueValuesEnum(_messages.Enum):
        """From how long ago in the past these intervals were observed.

    Values:
      DURATION_UNSPECIFIED: <no description>
      HOUR: <no description>
      MAX: From BfdSession object creation time.
      MINUTE: <no description>
    """
        DURATION_UNSPECIFIED = 0
        HOUR = 1
        MAX = 2
        MINUTE = 3

    class TypeValueValuesEnum(_messages.Enum):
        """The type of packets for which inter-packet intervals were computed.

    Values:
      LOOPBACK: Only applies to Echo packets. This shows the intervals between
        sending and receiving the same packet.
      RECEIVE: Intervals between received packets.
      TRANSMIT: Intervals between transmitted packets.
      TYPE_UNSPECIFIED: <no description>
    """
        LOOPBACK = 0
        RECEIVE = 1
        TRANSMIT = 2
        TYPE_UNSPECIFIED = 3
    avgMs = _messages.IntegerField(1)
    duration = _messages.EnumField('DurationValueValuesEnum', 2)
    maxMs = _messages.IntegerField(3)
    minMs = _messages.IntegerField(4)
    numIntervals = _messages.IntegerField(5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)