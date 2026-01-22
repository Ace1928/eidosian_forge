import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
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