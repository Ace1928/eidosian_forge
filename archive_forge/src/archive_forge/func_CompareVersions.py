from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import binascii
import codecs
import os
import sys
import io
import re
import locale
import collections
import random
import six
import string
from six.moves import urllib
from six.moves import range
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.utils.constants import UTF8
from gslib.utils.constants import WINDOWS_1252
from gslib.utils.system_util import IS_CP1252
def CompareVersions(first, second):
    """Compares the first and second gsutil version strings.

  For example, 3.33 > 3.7, and 4.1 is a greater major version than 3.33.
  Does not handle multiple periods (e.g. 3.3.4) or complicated suffixes
  (e.g., 3.3RC4 vs. 3.3RC5). A version string with a suffix is treated as
  less than its non-suffix counterpart (e.g. 3.32 > 3.32pre).

  Args:
    first: First gsutil version string.
    second: Second gsutil version string.

  Returns:
    (g, m):
       g is True if first known to be greater than second, else False.
       m is True if first known to be greater by at least 1 major version,
         else False.
  """
    m1 = VERSION_MATCHER().match(str(first))
    m2 = VERSION_MATCHER().match(str(second))
    if not m1 or not m2:
        return (False, False)
    major_ver1 = int(m1.group('maj'))
    minor_ver1 = int(m1.group('min')) if m1.group('min') else 0
    suffix_ver1 = m1.group('suffix')
    major_ver2 = int(m2.group('maj'))
    minor_ver2 = int(m2.group('min')) if m2.group('min') else 0
    suffix_ver2 = m2.group('suffix')
    if major_ver1 > major_ver2:
        return (True, True)
    elif major_ver1 == major_ver2:
        if minor_ver1 > minor_ver2:
            return (True, False)
        elif minor_ver1 == minor_ver2:
            return (bool(suffix_ver2) and (not suffix_ver1), False)
    return (False, False)