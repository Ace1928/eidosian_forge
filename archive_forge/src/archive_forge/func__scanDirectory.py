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
def _scanDirectory(self, dirIter, f):
    while len(f) < 250:
        try:
            info = next(dirIter)
        except StopIteration:
            if not f:
                raise EOFError
            return f
        if isinstance(info, defer.Deferred):
            info.addCallback(self._cbScanDirectory, dirIter, f)
            return
        else:
            f.append(info)
    return f