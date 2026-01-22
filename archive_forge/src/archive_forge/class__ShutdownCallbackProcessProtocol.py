import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
class _ShutdownCallbackProcessProtocol(ProcessProtocol):
    """
    An L{IProcessProtocol} which fires a Deferred when the process it is
    associated with ends.

    @ivar received: A C{dict} mapping file descriptors to lists of bytes
        received from the child process on those file descriptors.
    """

    def __init__(self, whenFinished):
        self.whenFinished = whenFinished
        self.received = {}

    def childDataReceived(self, fd, bytes):
        self.received.setdefault(fd, []).append(bytes)

    def processEnded(self, reason):
        self.whenFinished.callback(None)