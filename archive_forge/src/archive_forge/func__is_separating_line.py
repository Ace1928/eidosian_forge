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
def _is_separating_line(row):
    row_type = type(row)
    is_sl = (row_type == list or row_type == str) and (len(row) >= 1 and row[0] == SEPARATING_LINE or (len(row) >= 2 and row[1] == SEPARATING_LINE))
    return is_sl