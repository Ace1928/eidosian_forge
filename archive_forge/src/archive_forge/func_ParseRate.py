from __future__ import absolute_import
from __future__ import unicode_literals
import os
def ParseRate(rate):
    """Parses a rate string in the form `number/unit`, or the literal `0`.

  The unit is one of `s` (seconds), `m` (minutes), `h` (hours) or `d` (days).

  Args:
    rate: The string that contains the rate.

  Returns:
    A floating point number that represents the `rate/second`.

  Raises:
    MalformedQueueConfiguration: If the rate is invalid.
  """
    if rate == '0':
        return 0.0
    elements = rate.split('/')
    if len(elements) != 2:
        raise MalformedQueueConfiguration('Rate "%s" is invalid.' % rate)
    number, unit = elements
    try:
        number = float(number)
    except ValueError:
        raise MalformedQueueConfiguration('Rate "%s" is invalid: "%s" is not a number.' % (rate, number))
    if unit not in 'smhd':
        raise MalformedQueueConfiguration('Rate "%s" is invalid: "%s" is not one of s, m, h, d.' % (rate, unit))
    if unit == 's':
        return number
    if unit == 'm':
        return number / 60
    if unit == 'h':
        return number / (60 * 60)
    if unit == 'd':
        return number / (24 * 60 * 60)