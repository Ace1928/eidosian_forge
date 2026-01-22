import re
from numba.core import types
def _split_mangled_ident(mangled):
    """
    Returns `(head, tail)` where `head` is the `<len> + <name>` encoded
    identifier and `tail` is the remaining.
    """
    ct = int(mangled)
    ctlen = len(str(ct))
    at = ctlen + ct
    return (mangled[:at], mangled[at:])