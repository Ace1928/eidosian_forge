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
class PotentialZombieWarningTests(TestCase):
    """
    Tests for L{twisted.internet.error.PotentialZombieWarning}.
    """

    def test_deprecated(self):
        """
        Accessing L{PotentialZombieWarning} via the
        I{PotentialZombieWarning} attribute of L{twisted.internet.error}
        results in a deprecation warning being emitted.
        """
        from twisted.internet import error
        error.PotentialZombieWarning
        warnings = self.flushWarnings([self.test_deprecated])
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(warnings[0]['message'], 'twisted.internet.error.PotentialZombieWarning was deprecated in Twisted 10.0.0: There is no longer any potential for zombie process.')
        self.assertEqual(len(warnings), 1)