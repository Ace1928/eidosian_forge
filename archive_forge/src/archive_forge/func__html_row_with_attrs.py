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
def _html_row_with_attrs(celltag, unsafe, cell_values, colwidths, colaligns):
    alignment = {'left': '', 'right': ' style="text-align: right;"', 'center': ' style="text-align: center;"', 'decimal': ' style="text-align: right;"'}
    if unsafe:
        values_with_attrs = ['<{0}{1}>{2}</{0}>'.format(celltag, alignment.get(a, ''), c) for c, a in zip(cell_values, colaligns)]
    else:
        values_with_attrs = ['<{0}{1}>{2}</{0}>'.format(celltag, alignment.get(a, ''), htmlescape(c)) for c, a in zip(cell_values, colaligns)]
    rowhtml = '<tr>{}</tr>'.format(''.join(values_with_attrs).rstrip())
    if celltag == 'th':
        rowhtml = f'<table>\n<thead>\n{rowhtml}\n</thead>\n<tbody>'
    return rowhtml