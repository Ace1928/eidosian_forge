from __future__ import print_function, unicode_literals
from functools import update_wrapper
import six
import pybtex.io
from pybtex.bibtex import utils
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.names import format_name as format_bibtex_name
from pybtex.errors import report_error
from pybtex.utils import memoize
@builtin('add.period$')
def add_period(i):
    s = i.pop()
    if s and (not s.rstrip('}')[-1] in '.?!'):
        s += '.'
    i.push(s)