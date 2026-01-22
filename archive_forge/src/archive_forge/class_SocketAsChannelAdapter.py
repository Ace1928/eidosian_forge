import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
class SocketAsChannelAdapter:
    """Simple wrapper for a socket that pretends to be a paramiko Channel."""

    def __init__(self, sock):
        self.__socket = sock

    def get_name(self):
        return 'bzr SocketAsChannelAdapter'

    def send(self, data):
        return self.__socket.send(data)

    def recv(self, n):
        try:
            return self.__socket.recv(n)
        except OSError as e:
            if e.args[0] in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.EBADF):
                return ''
            raise

    def recv_ready(self):
        return True

    def close(self):
        self.__socket.close()