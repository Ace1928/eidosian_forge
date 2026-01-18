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
@onlyOnPOSIX
def test_registerReapProcessHandler(self):
    process.registerReapProcessHandler(None, None)
    [error] = self.flushLoggedErrors()
    self.assertEqual(type(error.value), self.expected_type, 'Wrong error type logged')
    self.assertEqual(str(error.value), self.expected_message, 'Wrong error message logged')