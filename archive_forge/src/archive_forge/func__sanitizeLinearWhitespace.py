from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def _sanitizeLinearWhitespace(headerComponent: bytes) -> bytes:
    """
    Replace linear whitespace (C{\\n}, C{\\r\\n}, C{\\r}) in a header key
    or value with a single space.

    @param headerComponent: The header key or value to sanitize.

    @return: The sanitized header key or value.
    """
    return b' '.join(headerComponent.splitlines())