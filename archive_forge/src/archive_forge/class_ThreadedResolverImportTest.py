import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import (
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
@skipIfNoNetwork
@unittest.skipIf(sys.platform == 'win32', 'preexec_fn not available on win32')
class ThreadedResolverImportTest(unittest.TestCase):

    def test_import(self):
        TIMEOUT = 5
        command = [sys.executable, '-c', 'import tornado.test.resolve_test_helper']
        start = time.time()
        popen = Popen(command, preexec_fn=lambda: signal.alarm(TIMEOUT))
        while time.time() - start < TIMEOUT:
            return_code = popen.poll()
            if return_code is not None:
                self.assertEqual(0, return_code)
                return
            time.sleep(0.05)
        self.fail('import timed out')