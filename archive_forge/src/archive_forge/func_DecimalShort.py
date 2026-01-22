from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def DecimalShort(num):
    """Creates a shorter string version for a given number of objects.

  Args:
    num: The number of objects to be shortened.
  Returns:
    shortened string version for this number. It takes the largest
    scale (thousand, million or billion) smaller than the number and divides it
    by that scale, indicated by a suffix with one decimal place. This will thus
    create a string of at most 6 characters, assuming num < 10^18.
    Example: 123456789 => 123.4m
  """
    for divisor_exp, suffix in reversed(_EXP_TEN_STRING):
        if num >= 10 ** divisor_exp:
            quotient = '%.1lf' % (float(num) / 10 ** divisor_exp)
            return quotient + suffix
    return str(num)