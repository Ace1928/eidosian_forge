from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def _process_token(cls, token):
    """Preprocess the token component of a token definition."""
    assert type(token) is _TokenType or callable(token), 'token type must be simple type or callable, not %r' % (token,)
    return token