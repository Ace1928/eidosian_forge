from __future__ import print_function
import re
from sqlparse import tokens as T
from sqlparse.compat import string_types, text_type, unicode_compatible
from sqlparse.utils import imt, remove_quotes
def token_index(self, token, start=0):
    """Return list index of token."""
    start = start if isinstance(start, int) else self.token_index(start)
    return start + self.tokens[start:].index(token)