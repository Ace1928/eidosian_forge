import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _is_name_start(css, pos):
    """Return true if the given character is a name-start code point."""
    c = css[pos]
    return c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_' or ord(c) > 127