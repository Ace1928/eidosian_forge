import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _consume_escape(css, pos):
    """Return (unescaped_char, new_pos).

    Assumes a valid escape: pos is just after '\\' and not followed by '\\n'.

    """
    hex_match = _HEX_ESCAPE_RE.match(css, pos)
    if hex_match:
        codepoint = int(hex_match.group(1), 16)
        return (chr(codepoint) if 0 < codepoint <= sys.maxunicode else '�', hex_match.end())
    elif pos < len(css):
        return (css[pos], pos + 1)
    else:
        return ('�', pos)