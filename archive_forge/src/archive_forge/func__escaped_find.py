import re
import unittest
from typing import (
from . import (
@staticmethod
def _escaped_find(regex: str, s: str, start: int, in_regex: bool) -> Tuple[str, int, int]:
    """Find the next slash in {s} after {start} that is not preceded by a backslash.

        If we find an escaped slash, add everything up to and including it to regex,
        updating {start}. {start} therefore serves two purposes, tells us where to start
        looking for the next thing, and also tells us where in {s} we have already
        added things to {regex}

        {in_regex} specifies whether we are currently searching in a regex, we behave
        differently if we are or if we aren't.
        """
    while True:
        pos = s.find('/', start)
        if pos == -1:
            break
        elif pos == 0:
            break
        elif s[pos - 1:pos] == '\\':
            if in_regex:
                regex += s[start:pos - 1]
                regex += s[pos]
            else:
                regex += re.escape(s[start:pos - 1])
                regex += re.escape(s[pos])
            start = pos + 1
        else:
            break
    return (regex, pos, start)