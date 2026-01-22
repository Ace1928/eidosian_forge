import re
import sys
from webencodings import ascii_lower
from .ast import (
from .serializer import serialize_string_value, serialize_url
def _consume_url(css, pos):
    """Return (unescaped_url, new_pos)

    The given pos is assumed to be just after the '(' of 'url('.

    """
    error = None
    length = len(css)
    while css.startswith((' ', '\n', '\t'), pos):
        pos += 1
    if pos >= length:
        return ('', pos, ('eof-in-url', 'EOF in URL'))
    c = css[pos]
    if c in ('"', "'"):
        value, pos, error = _consume_quoted_string(css, pos)
    elif c == ')':
        return ('', pos + 1, error)
    else:
        chunks = []
        start_pos = pos
        while 1:
            if pos >= length:
                chunks.append(css[start_pos:pos])
                return (''.join(chunks), pos, ('eof-in-url', 'EOF in URL'))
            c = css[pos]
            if c == ')':
                chunks.append(css[start_pos:pos])
                pos += 1
                return (''.join(chunks), pos, error)
            elif c in ' \n\t':
                chunks.append(css[start_pos:pos])
                value = ''.join(chunks)
                pos += 1
                break
            elif c == '\\' and (not css.startswith('\\\n', pos)):
                chunks.append(css[start_pos:pos])
                c, pos = _consume_escape(css, pos + 1)
                chunks.append(c)
                start_pos = pos
            elif c in '"\'(\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f':
                value = None
                pos += 1
                break
            else:
                pos += 1
    if value is not None:
        while css.startswith((' ', '\n', '\t'), pos):
            pos += 1
        if pos < length:
            if css[pos] == ')':
                return (value, pos + 1, error)
        else:
            if error is None:
                error = ('eof-in-url', 'EOF in URL')
            return (value, pos, error)
    while pos < length:
        if css.startswith('\\)', pos):
            pos += 2
        elif css[pos] == ')':
            pos += 1
            break
        else:
            pos += 1
    return (None, pos, ('bad-url', 'bad URL token'))