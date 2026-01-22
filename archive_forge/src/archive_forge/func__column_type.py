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
def _column_type(strings, has_invisible=True, numparse=True):
    """The least generic type all column values are convertible to.

    >>> _column_type([True, False]) is bool
    True
    >>> _column_type(["1", "2"]) is int
    True
    >>> _column_type(["1", "2.3"]) is float
    True
    >>> _column_type(["1", "2.3", "four"]) is str
    True
    >>> _column_type(["four", 'пять']) is str
    True
    >>> _column_type([None, "brux"]) is str
    True
    >>> _column_type([1, 2, None]) is int
    True
    >>> import datetime as dt
    >>> _column_type([dt.datetime(1991,2,19), dt.time(17,35)]) is str
    True

    """
    types = [_type(s, has_invisible, numparse) for s in strings]
    return reduce(_more_generic, types, bool)