from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def token_next_by(self, i=None, m=None, t=None, idx=-1, end=None):
    funcs = lambda tk: imt(tk, i, m, t)
    idx += 1
    return self._token_matching(funcs, idx, end)