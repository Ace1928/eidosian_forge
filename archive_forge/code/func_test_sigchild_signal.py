import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from tornado.httpclient import HTTPClient, HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.log import gen_log
from tornado.process import fork_processes, task_id, Subprocess
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import bind_unused_port, ExpectLog, AsyncTestCase, gen_test
from tornado.test.util import skipIfNonUnix
from tornado.web import RequestHandler, Application
def test_sigchild_signal(self):
    Subprocess.initialize()
    self.addCleanup(Subprocess.uninitialize)
    subproc = Subprocess([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=Subprocess.STREAM)
    self.addCleanup(subproc.stdout.close)
    subproc.set_exit_callback(self.stop)
    time.sleep(0.1)
    os.kill(subproc.pid, signal.SIGTERM)
    try:
        ret = self.wait()
    except AssertionError:
        fut = subproc.stdout.read_until_close()
        fut.add_done_callback(lambda f: self.stop())
        try:
            self.wait()
        except AssertionError:
            raise AssertionError('subprocess failed to terminate')
        else:
            raise AssertionError('subprocess closed stdout but failed to get termination signal')
    self.assertEqual(subproc.returncode, ret)
    self.assertEqual(ret, -signal.SIGTERM)