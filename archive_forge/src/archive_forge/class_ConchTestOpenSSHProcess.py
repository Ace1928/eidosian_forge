import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
class ConchTestOpenSSHProcess(protocol.ProcessProtocol):
    """
    Test protocol for launching an OpenSSH client process.

    @ivar deferred: Set by whatever uses this object. Accessed using
    L{_getDeferred}, which destroys the value so the Deferred is not
    fired twice. Fires when the process is terminated.
    """
    deferred = None
    buf = b''
    problems = b''

    def _getDeferred(self):
        d, self.deferred = (self.deferred, None)
        return d

    def outReceived(self, data):
        self.buf += data

    def errReceived(self, data):
        self.problems += data

    def processEnded(self, reason):
        """
        Called when the process has ended.

        @param reason: a Failure giving the reason for the process' end.
        """
        if reason.value.exitCode != 0:
            self._getDeferred().errback(ConchError('exit code was not 0: {} ({})'.format(reason.value.exitCode, self.problems.decode('charmap'))))
        else:
            buf = self.buf.replace(b'\r\n', b'\n')
            self._getDeferred().callback(buf)