from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _isnumber(string):
    """
    >>> _isnumber("123.45")
    True
    >>> _isnumber("123")
    True
    >>> _isnumber("spam")
    False
    >>> _isnumber("123e45678")
    False
    >>> _isnumber("inf")
    True
    """
    if not _isconvertible(float, string):
        return False
    elif isinstance(string, (str, bytes)) and (math.isinf(float(string)) or math.isnan(float(string))):
        return string.lower() in ['inf', '-inf', 'nan']
    return True