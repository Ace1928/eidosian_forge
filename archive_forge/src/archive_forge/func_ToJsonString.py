import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def ToJsonString(self):
    """Converts Duration to string format.

    Returns:
      A string converted from self. The string format will contains
      3, 6, or 9 fractional digits depending on the precision required to
      represent the exact Duration value. For example: "1s", "1.010s",
      "1.000000100s", "-3.100s"
    """
    _CheckDurationValid(self.seconds, self.nanos)
    if self.seconds < 0 or self.nanos < 0:
        result = '-'
        seconds = -self.seconds + int((0 - self.nanos) // 1000000000.0)
        nanos = (0 - self.nanos) % 1000000000.0
    else:
        result = ''
        seconds = self.seconds + int(self.nanos // 1000000000.0)
        nanos = self.nanos % 1000000000.0
    result += '%d' % seconds
    if nanos % 1000000000.0 == 0:
        return result + 's'
    if nanos % 1000000.0 == 0:
        return result + '.%03ds' % (nanos / 1000000.0)
    if nanos % 1000.0 == 0:
        return result + '.%06ds' % (nanos / 1000.0)
    return result + '.%09ds' % nanos