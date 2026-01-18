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
def test_sigchild(self):
    Subprocess.initialize()
    self.addCleanup(Subprocess.uninitialize)
    subproc = Subprocess([sys.executable, '-c', 'pass'])
    subproc.set_exit_callback(self.stop)
    ret = self.wait()
    self.assertEqual(ret, 0)
    self.assertEqual(subproc.returncode, ret)