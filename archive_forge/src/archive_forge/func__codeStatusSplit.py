import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _codeStatusSplit(line):
    """
    Parse the first line of a multi-line server response.

    @type line: L{bytes}
    @param line: The first line of a multi-line server response.

    @rtype: 2-tuple of (0) L{bytes}, (1) L{bytes}
    @return: The status indicator and the rest of the server response.
    """
    parts = line.split(b' ', 1)
    if len(parts) == 1:
        return (parts[0], b'')
    return parts