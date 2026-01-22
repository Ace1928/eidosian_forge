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
def _isnumber_with_thousands_separator(string):
    """
    >>> _isnumber_with_thousands_separator(".")
    False
    >>> _isnumber_with_thousands_separator("1")
    True
    >>> _isnumber_with_thousands_separator("1.")
    True
    >>> _isnumber_with_thousands_separator(".1")
    True
    >>> _isnumber_with_thousands_separator("1000")
    False
    >>> _isnumber_with_thousands_separator("1,000")
    True
    >>> _isnumber_with_thousands_separator("1,0000")
    False
    >>> _isnumber_with_thousands_separator("1,000.1234")
    True
    >>> _isnumber_with_thousands_separator(b"1,000.1234")
    True
    >>> _isnumber_with_thousands_separator("+1,000.1234")
    True
    >>> _isnumber_with_thousands_separator("-1,000.1234")
    True
    """
    try:
        string = string.decode()
    except (UnicodeDecodeError, AttributeError):
        pass
    return bool(re.match(_float_with_thousands_separators, string))