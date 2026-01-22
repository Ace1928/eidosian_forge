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
def _is_multiline(s):
    if isinstance(s, str):
        return bool(re.search(_multiline_codes, s))
    else:
        return bool(re.search(_multiline_codes_bytes, s))