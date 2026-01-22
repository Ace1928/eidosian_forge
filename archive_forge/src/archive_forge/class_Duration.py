import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
class Duration(object):
    """Class for Duration message type."""
    __slots__ = ()

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

    def FromJsonString(self, value):
        """Converts a string to Duration.

    Args:
      value: A string to be converted. The string must end with 's'. Any
          fractional digits (or none) are accepted as long as they fit into
          precision. For example: "1s", "1.01s", "1.0000001s", "-3.100s

    Raises:
      ValueError: On parsing problems.
    """
        if not isinstance(value, str):
            raise ValueError('Duration JSON value not a string: {!r}'.format(value))
        if len(value) < 1 or value[-1] != 's':
            raise ValueError('Duration must end with letter "s": {0}.'.format(value))
        try:
            pos = value.find('.')
            if pos == -1:
                seconds = int(value[:-1])
                nanos = 0
            else:
                seconds = int(value[:pos])
                if value[0] == '-':
                    nanos = int(round(float('-0{0}'.format(value[pos:-1])) * 1000000000.0))
                else:
                    nanos = int(round(float('0{0}'.format(value[pos:-1])) * 1000000000.0))
            _CheckDurationValid(seconds, nanos)
            self.seconds = seconds
            self.nanos = nanos
        except ValueError as e:
            raise ValueError("Couldn't parse duration: {0} : {1}.".format(value, e))

    def ToNanoseconds(self):
        """Converts a Duration to nanoseconds."""
        return self.seconds * _NANOS_PER_SECOND + self.nanos

    def ToMicroseconds(self):
        """Converts a Duration to microseconds."""
        micros = _RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND)
        return self.seconds * _MICROS_PER_SECOND + micros

    def ToMilliseconds(self):
        """Converts a Duration to milliseconds."""
        millis = _RoundTowardZero(self.nanos, _NANOS_PER_MILLISECOND)
        return self.seconds * _MILLIS_PER_SECOND + millis

    def ToSeconds(self):
        """Converts a Duration to seconds."""
        return self.seconds

    def FromNanoseconds(self, nanos):
        """Converts nanoseconds to Duration."""
        self._NormalizeDuration(nanos // _NANOS_PER_SECOND, nanos % _NANOS_PER_SECOND)

    def FromMicroseconds(self, micros):
        """Converts microseconds to Duration."""
        self._NormalizeDuration(micros // _MICROS_PER_SECOND, micros % _MICROS_PER_SECOND * _NANOS_PER_MICROSECOND)

    def FromMilliseconds(self, millis):
        """Converts milliseconds to Duration."""
        self._NormalizeDuration(millis // _MILLIS_PER_SECOND, millis % _MILLIS_PER_SECOND * _NANOS_PER_MILLISECOND)

    def FromSeconds(self, seconds):
        """Converts seconds to Duration."""
        self.seconds = seconds
        self.nanos = 0

    def ToTimedelta(self):
        """Converts Duration to timedelta."""
        return datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))

    def FromTimedelta(self, td):
        """Converts timedelta to Duration."""
        self._NormalizeDuration(td.seconds + td.days * _SECONDS_PER_DAY, td.microseconds * _NANOS_PER_MICROSECOND)

    def _NormalizeDuration(self, seconds, nanos):
        """Set Duration by seconds and nanos."""
        if seconds < 0 and nanos > 0:
            seconds += 1
            nanos -= _NANOS_PER_SECOND
        self.seconds = seconds
        self.nanos = nanos