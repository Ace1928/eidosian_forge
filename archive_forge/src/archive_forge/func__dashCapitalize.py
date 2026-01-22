from collections.abc import Sequence as _Sequence
from typing import (
from twisted.python.compat import cmp, comparable
def _dashCapitalize(name: bytes) -> bytes:
    """
    Return a byte string which is capitalized using '-' as a word separator.

    @param name: The name of the header to capitalize.

    @return: The given header capitalized using '-' as a word separator.
    """
    return b'-'.join([word.capitalize() for word in name.split(b'-')])