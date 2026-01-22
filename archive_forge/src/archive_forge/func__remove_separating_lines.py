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
def _remove_separating_lines(rows):
    if type(rows) == list:
        separating_lines = []
        sans_rows = []
        for index, row in enumerate(rows):
            if _is_separating_line(row):
                separating_lines.append(index)
            else:
                sans_rows.append(row)
        return (sans_rows, separating_lines)
    else:
        return (rows, None)