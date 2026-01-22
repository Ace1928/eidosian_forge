import argparse
import ast
import re
import sys
def _parse_interval(interval_str):
    """Convert a human-readable interval to a tuple of start and end value.

  Args:
    interval_str: (`str`) A human-readable str representing an interval
      (e.g., "[1M, 2M]", "<100k", ">100ms"). The items following the ">", "<",
      ">=" and "<=" signs have to start with a number (e.g., 3.0, -2, .98).
      The same requirement applies to the items in the parentheses or brackets.

  Returns:
    Interval object where start or end can be None
    if the range is specified as "<N" or ">N" respectively.

  Raises:
    ValueError: if the input is not valid.
  """
    interval_str = interval_str.strip()
    if interval_str.startswith('<='):
        if _NUMBER_PATTERN.match(interval_str[2:].strip()):
            return Interval(start=None, start_included=False, end=interval_str[2:].strip(), end_included=True)
        else:
            raise ValueError("Invalid value string after <= in '%s'" % interval_str)
    if interval_str.startswith('<'):
        if _NUMBER_PATTERN.match(interval_str[1:].strip()):
            return Interval(start=None, start_included=False, end=interval_str[1:].strip(), end_included=False)
        else:
            raise ValueError("Invalid value string after < in '%s'" % interval_str)
    if interval_str.startswith('>='):
        if _NUMBER_PATTERN.match(interval_str[2:].strip()):
            return Interval(start=interval_str[2:].strip(), start_included=True, end=None, end_included=False)
        else:
            raise ValueError("Invalid value string after >= in '%s'" % interval_str)
    if interval_str.startswith('>'):
        if _NUMBER_PATTERN.match(interval_str[1:].strip()):
            return Interval(start=interval_str[1:].strip(), start_included=False, end=None, end_included=False)
        else:
            raise ValueError("Invalid value string after > in '%s'" % interval_str)
    if not interval_str.startswith(('[', '(')) or not interval_str.endswith((']', ')')):
        raise ValueError('Invalid interval format: %s. Valid formats are: [min, max], (min, max), <max, >min' % interval_str)
    interval = interval_str[1:-1].split(',')
    if len(interval) != 2:
        raise ValueError('Incorrect interval format: %s. Interval should specify two values: [min, max] or (min, max).' % interval_str)
    start_item = interval[0].strip()
    if not _NUMBER_PATTERN.match(start_item):
        raise ValueError("Invalid first item in interval: '%s'" % start_item)
    end_item = interval[1].strip()
    if not _NUMBER_PATTERN.match(end_item):
        raise ValueError("Invalid second item in interval: '%s'" % end_item)
    return Interval(start=start_item, start_included=interval_str[0] == '[', end=end_item, end_included=interval_str[-1] == ']')