import sys
import types
from cgi import parse_header
def text_(s, encoding='latin-1', errors='strict'):
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    return s