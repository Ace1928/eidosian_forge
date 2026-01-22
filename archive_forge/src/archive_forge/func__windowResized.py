import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def _windowResized(self, *args):
    winsz = fcntl.ioctl(0, tty.TIOCGWINSZ, '12345678')
    winSize = struct.unpack('4H', winsz)
    newSize = (winSize[1], winSize[0], winSize[2], winSize[3])
    self.conn.sendRequest(self, b'window-change', struct.pack('!4L', *newSize))