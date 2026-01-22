import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
class WarningCheckerTestCase(TestCase):
    """
    A test case that will make sure that no warnings are left unchecked at the end of a test run.
    """

    def setUp(self):
        super().setUp()
        if os.environ.get('CI', '').lower() == 'true' and platform.isWindows():
            self.flushWarnings()

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            warnings = self.flushWarnings()
            if os.environ.get('CI', '').lower() == 'true' and platform.isWindows():
                return
            self.assertEqual(len(warnings), 0, f'Warnings found at the end of the test:\n{warnings}')