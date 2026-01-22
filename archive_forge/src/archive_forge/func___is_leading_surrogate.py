from __future__ import absolute_import
import functools
import json
import re
import sys
@staticmethod
def __is_leading_surrogate(c):
    """Returns true if 'c' is a Unicode code point for a leading
        surrogate."""
    return c >= 55296 and c <= 56319