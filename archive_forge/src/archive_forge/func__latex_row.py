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
def _latex_row(cell_values, colwidths, colaligns, escrules=LATEX_ESCAPE_RULES):

    def escape_char(c):
        return escrules.get(c, c)
    escaped_values = [''.join(map(escape_char, cell)) for cell in cell_values]
    rowfmt = DataRow('', '&', '\\\\')
    return _build_simple_row(escaped_values, rowfmt)