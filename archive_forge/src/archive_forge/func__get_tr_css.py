from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
def _get_tr_css(row, tr_style):
    if tr_style:
        if isinstance(tr_style, string_types):
            return tr_style
        elif callable(tr_style):
            return tr_style(row)
        else:
            raise ArgumentError('expected string or callable, got %r' % tr_style)
    return ''