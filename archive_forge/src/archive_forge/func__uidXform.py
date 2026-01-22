import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _uidXform(line):
    """
    Parse a line of the response to a UIDL command.

    The line from the UIDL response consists of a 1-based message number
    followed by a unique id.

    @type line: L{bytes}
    @param line: A non-initial line from the multi-line response to a UIDL
        command.

    @rtype: 2-L{tuple} of (0) L{int}, (1) L{bytes}
    @return: The 0-based index of the message and the unique identifier
        for the message.
    """
    index, uid = line.split(None, 1)
    return (int(index) - 1, uid)