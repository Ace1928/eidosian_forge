import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
@implementer(IReadFile)
class _FileReader:

    def __init__(self, fObj):
        self.fObj = fObj
        self._send = False

    def _close(self, passthrough):
        self._send = True
        self.fObj.close()
        return passthrough

    def send(self, consumer):
        assert not self._send, 'Can only call IReadFile.send *once* per instance'
        self._send = True
        d = basic.FileSender().beginFileTransfer(self.fObj, consumer)
        d.addBoth(self._close)
        return d