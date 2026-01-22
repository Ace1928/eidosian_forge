from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def FormatBinaryNumber(value, type_abbr='B', decimal_places=-1):
    """Returns a pretty string of a binary-base number with decimal precision.

  Args:
    value (float|int): A number.
    type_abbr (str): The optional type abbreviation suffix, validated but
      otherwise ignored.
    decimal_places (int): Number of decimal places to include of quotient for
      unit conversion. Does not allow rounding if -1. Will suffer float
      inaccuracy at high values.

  Returns:
    A formatted scaled value string.
  """
    for suffix, size in reversed(sorted(six.iteritems(_BINARY_UNITS), key=lambda value: (value[1], value[0]))):
        if size <= value:
            if decimal_places == -1 and value % size:
                continue
            scaled_value = value / size
            precision = max(decimal_places, 0)
            return '{:.{precision}f}{}{}'.format(scaled_value, suffix, type_abbr, precision=precision)
    return '{}{}'.format(value, type_abbr)