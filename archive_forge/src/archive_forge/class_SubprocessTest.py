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
@skipIfNonUnix
class SubprocessTest(AsyncTestCase):

    def term_and_wait(self, subproc):
        subproc.proc.terminate()
        subproc.proc.wait()

    @gen_test
    def test_subprocess(self):
        if IOLoop.configured_class().__name__.endswith('LayeredTwistedIOLoop'):
            raise unittest.SkipTest('Subprocess tests not compatible with LayeredTwistedIOLoop')
        subproc = Subprocess([sys.executable, '-u', '-i'], stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: self.term_and_wait(subproc))
        self.addCleanup(subproc.stdout.close)
        self.addCleanup(subproc.stdin.close)
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.write(b"print('hello')\n")
        data = (yield subproc.stdout.read_until(b'\n'))
        self.assertEqual(data, b'hello\n')
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.write(b'raise SystemExit\n')
        data = (yield subproc.stdout.read_until_close())
        self.assertEqual(data, b'')

    @gen_test
    def test_close_stdin(self):
        subproc = Subprocess([sys.executable, '-u', '-i'], stdin=Subprocess.STREAM, stdout=Subprocess.STREAM, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: self.term_and_wait(subproc))
        yield subproc.stdout.read_until(b'>>> ')
        subproc.stdin.close()
        data = (yield subproc.stdout.read_until_close())
        self.assertEqual(data, b'\n')

    @gen_test
    def test_stderr(self):
        subproc = Subprocess([sys.executable, '-u', '-c', "import sys; sys.stderr.write('hello\\n')"], stderr=Subprocess.STREAM)
        self.addCleanup(lambda: self.term_and_wait(subproc))
        data = (yield subproc.stderr.read_until(b'\n'))
        self.assertEqual(data, b'hello\n')
        subproc.stderr.close()

    def test_sigchild(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'pass'])
        subproc.set_exit_callback(self.stop)
        ret = self.wait()
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

    @gen_test
    def test_sigchild_future(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'pass'])
        ret = (yield subproc.wait_for_exit())
        self.assertEqual(ret, 0)
        self.assertEqual(subproc.returncode, ret)

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

    @gen_test
    def test_wait_for_exit_raise(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'import sys; sys.exit(1)'])
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            yield subproc.wait_for_exit()
        self.assertEqual(cm.exception.returncode, 1)

    @gen_test
    def test_wait_for_exit_raise_disabled(self):
        Subprocess.initialize()
        self.addCleanup(Subprocess.uninitialize)
        subproc = Subprocess([sys.executable, '-c', 'import sys; sys.exit(1)'])
        ret = (yield subproc.wait_for_exit(raise_error=False))
        self.assertEqual(ret, 1)