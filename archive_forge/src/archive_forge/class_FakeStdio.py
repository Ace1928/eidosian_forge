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
class FakeStdio:
    """
    A fake for testing L{twisted.conch.scripts.conch.SSHSession.eofReceived} and
    L{twisted.conch.scripts.cftp.SSHSession.eofReceived}.

    @ivar writeConnLost: A flag which records whether L{loserWriteConnection}
        has been called.
    """
    writeConnLost = False

    def loseWriteConnection(self):
        """
        Record the call to loseWriteConnection.
        """
        self.writeConnLost = True