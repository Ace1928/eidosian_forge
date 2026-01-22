import os
import signal
import sys
import time
from breezy import debug, tests
class TestDebugOption(tests.TestCaseInTempDir):

    def test_dash_derror(self):
        """With -Derror, tracebacks are shown even for user errors"""
        out, err = self.run_bzr('-Derror branch nonexistent-location', retcode=3)
        self.assertContainsRe(err, 'Traceback \\(most recent call last\\)')

    def test_dash_dlock(self):
        self.run_bzr('-Dlock init foo')
        self.assertContainsRe(self.get_log(), 'lock_write')