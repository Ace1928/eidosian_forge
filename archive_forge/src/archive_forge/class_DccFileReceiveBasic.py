import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class DccFileReceiveBasic(protocol.Protocol, styles.Ephemeral):
    """
    Bare protocol to receive a Direct Client Connection SEND stream.

    This does enough to keep the other guy talking, but you'll want to extend
    my dataReceived method to *do* something with the data I get.

    @ivar bytesReceived: An integer representing the number of bytes of data
        received.
    @type bytesReceived: L{int}
    """
    bytesReceived = 0

    def __init__(self, resumeOffset=0):
        """
        @param resumeOffset: An integer representing the amount of bytes from
            where the transfer of data should be resumed.
        @type resumeOffset: L{int}
        """
        self.bytesReceived = resumeOffset
        self.resume = resumeOffset != 0

    def dataReceived(self, data):
        """
        See: L{protocol.Protocol.dataReceived}

        Warning: This just acknowledges to the remote host that the data has
        been received; it doesn't I{do} anything with the data, so you'll want
        to override this.
        """
        self.bytesReceived = self.bytesReceived + len(data)
        self.transport.write(struct.pack('!i', self.bytesReceived))