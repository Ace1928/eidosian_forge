import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _statXform(line):
    """
    Parse the response to a STAT command.

    @type line: L{bytes}
    @param line: The response from the server to a STAT command minus the
        status indicator.

    @rtype: 2-L{tuple} of (0) L{int}, (1) L{int}
    @return: The number of messages in the mailbox and the size of the mailbox.
    """
    numMsgs, totalSize = line.split(None, 1)
    return (int(numMsgs), int(totalSize))