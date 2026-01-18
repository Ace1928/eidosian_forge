from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def token_not_matching(self, funcs, idx):
    funcs = (funcs,) if not isinstance(funcs, (list, tuple)) else funcs
    funcs = [lambda tk: not func(tk) for func in funcs]
    return self._token_matching(funcs, idx)