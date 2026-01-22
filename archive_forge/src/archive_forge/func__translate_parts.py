import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
@functools.lru_cache(maxsize=100)
def _translate_parts(self, value):
    """HTML-escape a value and split it by newlines."""
    return value.translate(_escape_html_table).split('\n')