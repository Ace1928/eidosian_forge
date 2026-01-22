from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def HumanReadableToBytes(human_string):
    """Tries to convert a human-readable string to a number of bytes.

  Args:
    human_string: A string supplied by user, e.g. '1M', '3 GiB'.
  Returns:
    An integer containing the number of bytes.
  Raises:
    ValueError: on an invalid string.
  """
    human_string = human_string.lower()
    m = MATCH_HUMAN_BYTES.match(human_string)
    if m:
        num = float(m.group('num'))
        if m.group('suffix'):
            power = _EXP_STRINGS[SUFFIX_TO_SI[m.group('suffix')]][0]
            num *= 2.0 ** power
        num = int(round(num))
        return num
    raise ValueError('Invalid byte string specified: %s' % human_string)