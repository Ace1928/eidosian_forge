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
@implementer(IWriteFile)
class _FileWriter:

    def __init__(self, fObj):
        self.fObj = fObj
        self._receive = False

    def receive(self):
        assert not self._receive, 'Can only call IWriteFile.receive *once* per instance'
        self._receive = True
        return defer.succeed(FileConsumer(self.fObj))

    def close(self):
        return defer.succeed(None)