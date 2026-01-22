import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
class SFTPError(Exception):

    def __init__(self, errorCode, errorMessage, lang=''):
        Exception.__init__(self)
        self.code = errorCode
        self._message = errorMessage
        self.lang = lang

    @property
    def message(self):
        """
        A string received over the network that explains the error to a human.
        """
        return self._message

    def __str__(self) -> str:
        return f'SFTPError {self.code}: {self.message}'