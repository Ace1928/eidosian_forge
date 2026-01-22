from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def _get_repr_value(self):
    raw = text_type(self)
    if len(raw) > 7:
        raw = raw[:6] + '...'
    return re.sub('\\s+', ' ', raw)