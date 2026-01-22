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
@staticmethod
def _len(item):
    """Custom len that gets console column width for wide
        and non-wide characters as well as ignores color codes"""
    stripped = _strip_ansi(item)
    if wcwidth:
        return wcwidth.wcswidth(stripped)
    else:
        return len(stripped)