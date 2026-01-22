import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
class RootwrapDaemonTest(_FunctionalBase, testtools.TestCase):

    def assert_unpatched(self):
        if eventlet and eventlet.patcher.is_monkey_patched('socket'):
            self.fail('Standard library should not be patched by eventlet for this test')

    def setUp(self):
        self.assert_unpatched()
        super(RootwrapDaemonTest, self).setUp()
        daemon_log = io.BytesIO()
        p = mock.patch('oslo_rootwrap.subprocess.Popen', run_daemon.forwarding_popen(daemon_log))
        p.start()
        self.addCleanup(p.stop)
        client_log = io.StringIO()
        handler = logging.StreamHandler(client_log)
        log_format = run_daemon.log_format.replace('+', ' ')
        handler.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger('oslo_rootwrap')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, handler)

        @self.addCleanup
        def add_logs():
            self.addDetail('daemon_log', content.Content(content.UTF8_TEXT, lambda: [daemon_log.getvalue()]))
            self.addDetail('client_log', content.Content(content.UTF8_TEXT, lambda: [client_log.getvalue().encode('utf-8')]))
        self.client = client.Client([sys.executable, run_daemon.__file__, self.config_file])

        @self.addCleanup
        def finalize_client():
            if self.client._initialized:
                self.client._finalize()
        self.execute = self.client.execute

    def test_run_once(self):
        self._test_run_once(expect_byte=False)

    def test_run_with_stdin(self):
        self._test_run_with_stdin(expect_byte=False)

    def test_run_with_later_install_cmd(self):
        code, out, err = self.execute(['later_install_cmd'])
        self.assertEqual(cmd.RC_NOEXECFOUND, code)
        shutil.copy('/bin/echo', self.later_cmd)
        code, out, err = self.execute(['later_install_cmd'])
        self.assertEqual(0, code)

    def test_daemon_ressurection(self):
        self.execute(['cat'])
        os.kill(self.client._process.pid, signal.SIGTERM)
        self.test_run_once()

    def test_daemon_timeout(self):
        self.execute(['echo'])
        with mock.patch.object(self.client, '_restart') as restart:
            time.sleep(15)
            self.execute(['echo'])
            restart.assert_called_once()

    def _exec_thread(self, fifo_path):
        try:
            self._thread_res = self.execute(['sh', '-c', 'echo > "%s"; sleep 1; echo OK' % fifo_path])
        except Exception as e:
            self._thread_res = e

    def test_graceful_death(self):
        tmpdir = self.useFixture(fixtures.TempDir()).path
        fifo_path = os.path.join(tmpdir, 'fifo')
        os.mkfifo(fifo_path)
        self.execute(['cat'])
        t = threading.Thread(target=self._exec_thread, args=(fifo_path,))
        t.start()
        with open(fifo_path) as f:
            f.readline()
        os.kill(self.client._process.pid, signal.SIGTERM)
        t.join()
        if isinstance(self._thread_res, Exception):
            raise self._thread_res
        code, out, err = self._thread_res
        self.assertEqual(0, code)
        self.assertEqual('OK\n', out)
        self.assertEqual('', err)

    @contextlib.contextmanager
    def _test_daemon_cleanup(self):
        self.execute(['cat'])
        socket_path = self.client._manager._address
        yield
        process = self.client._process
        stop = threading.Event()

        def sleep_kill():
            stop.wait(1)
            if not stop.is_set():
                os.kill(process.pid, signal.SIGKILL)
        threading.Thread(target=sleep_kill).start()
        self.client._process.wait()
        stop.set()
        self.assertNotEqual(-signal.SIGKILL, process.returncode, "Server haven't stopped in one second")
        self.assertFalse(os.path.exists(socket_path), "Server didn't remove its temporary directory")

    def test_daemon_cleanup_client(self):
        with self._test_daemon_cleanup():
            self.client._finalize()

    def test_daemon_cleanup_signal(self):
        with self._test_daemon_cleanup():
            os.kill(self.client._process.pid, signal.SIGTERM)