from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_xml_footer(style, epilogue, rows, root):
    nested = -1 if style in ('attribute', 'name') else -2
    tbd = _build_nesting(rows, True, nested)
    tab = _build_nesting(root, True, 0)
    pre = epilogue + '\n' if epilogue and root else ''
    pos = '\n' + epilogue if epilogue and (not root) else ''
    return pre + tbd + tab + pos