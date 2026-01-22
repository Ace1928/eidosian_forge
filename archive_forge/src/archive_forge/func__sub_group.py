import re
from . import lazy_regex
from .trace import mutter, warning
def _sub_group(m):
    if m[1] in ('!', '^'):
        return '[^' + _sub_named(m[2:-1]) + ']'
    return '[' + _sub_named(m[1:-1]) + ']'