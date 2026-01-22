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
def _padleft(width, s):
    """Flush right.

    >>> _padleft(6, 'яйца') == '  яйца'
    True

    """
    fmt = '{0:>%ds}' % width
    return fmt.format(s)