import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _listXform(line):
    """
    Parse a line of the response to a LIST command.

    The line from the LIST response consists of a 1-based message number
    followed by a size.

    @type line: L{bytes}
    @param line: A non-initial line from the multi-line response to a LIST
        command.

    @rtype: 2-L{tuple} of (0) L{int}, (1) L{int}
    @return: The 0-based index of the message and the size of the message.
    """
    index, size = line.split(None, 1)
    return (int(index) - 1, int(size))