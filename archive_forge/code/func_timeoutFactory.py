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
def timeoutFactory(self):
    log.msg('timed out waiting for DTP connection')
    if self._state is not self._IN_PROGRESS:
        return
    self._state = self._FAILED
    d = self.deferred
    self.deferred = None
    d.errback(PortConnectionError(defer.TimeoutError('DTPFactory timeout')))