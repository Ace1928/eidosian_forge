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
def _strip_ansi(s):
    """Remove ANSI escape sequences, both CSI (color codes, etc) and OSC hyperlinks.

    CSI sequences are simply removed from the output, while OSC hyperlinks are replaced
    with the link text. Note: it may be desirable to show the URI instead but this is not
    supported.

    >>> repr(_strip_ansi('\\x1B]8;;https://example.com\\x1B\\\\This is a link\\x1B]8;;\\x1B\\\\'))
    "'This is a link'"

    >>> repr(_strip_ansi('\\x1b[31mred\\x1b[0m text'))
    "'red text'"

    """
    if isinstance(s, str):
        return _ansi_codes.sub('\\4', s)
    else:
        return _ansi_codes_bytes.sub('\\4', s)