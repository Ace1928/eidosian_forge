from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_cols(style, props, path, is_value):
    is_header = not is_value
    if style == 'tag' or is_header:
        return _build_cols_inline(props, path, is_value, True)
    if style == 'name':
        return _build_cols_inline(props, path, is_value, False)
    if style == 'attribute':
        return _build_cols_attribs(props, path)
    return style