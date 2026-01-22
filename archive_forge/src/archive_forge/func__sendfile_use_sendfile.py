fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def _sendfile_use_sendfile(self, file, offset=0, count=None):
    raise _GiveupOnSendfile('os.sendfile() not available on this platform')