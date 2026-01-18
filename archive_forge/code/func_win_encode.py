import locale
import os
import sys
from gitdb.utils.encoding import force_bytes, force_text  # noqa: F401  # @UnusedImport
from typing import (  # noqa: F401
def win_encode(s: Optional[AnyStr]) -> Optional[bytes]:
    """Encode Unicode strings for process arguments on Windows."""
    if isinstance(s, str):
        return s.encode(locale.getpreferredencoding(False))
    elif isinstance(s, bytes):
        return s
    elif s is not None:
        raise TypeError('Expected bytes or text, but got %r' % (s,))
    return None