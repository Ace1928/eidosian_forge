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
def _latex_line_begin_tabular(colwidths, colaligns, booktabs=False, longtable=False):
    alignment = {'left': 'l', 'right': 'r', 'center': 'c', 'decimal': 'r'}
    tabular_columns_fmt = ''.join([alignment.get(a, 'l') for a in colaligns])
    return '\n'.join([('\\begin{tabular}{' if not longtable else '\\begin{longtable}{') + tabular_columns_fmt + '}', '\\toprule' if booktabs else '\\hline'])