import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
class EventletServerProcessLauncherTest(base.ServiceBaseTestCase):

    def setUp(self):
        super(EventletServerProcessLauncherTest, self).setUp()
        self.conf(args=[], default_config_files=[])
        self.addCleanup(self.conf.reset)
        self.workers = 3

    def run_server(self):
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=eventlet_service.run, args=(queue,), kwargs={'workers': self.workers, 'process_time': 5})
        proc.start()
        port = queue.get()
        conn = socket.create_connection(('127.0.0.1', port))
        conn.sendall(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
        time.sleep(1)
        return (proc, conn)

    def test_shuts_down_on_sigint_when_client_connected(self):
        proc, conn = self.run_server()
        self.assertTrue(proc.is_alive())
        os.kill(proc.pid, signal.SIGINT)
        proc.join()
        conn.close()

    def test_graceful_shuts_down_on_sigterm_when_client_connected(self):
        self.config(graceful_shutdown_timeout=7)
        proc, conn = self.run_server()
        os.kill(proc.pid, signal.SIGTERM)
        time.sleep(1)
        self.assertTrue(proc.is_alive())
        conn.close()
        proc.join()

    def test_graceful_stop_with_exceeded_graceful_shutdown_timeout(self):
        graceful_shutdown_timeout = 4
        self.config(graceful_shutdown_timeout=graceful_shutdown_timeout)
        proc, conn = self.run_server()
        time_before = time.time()
        os.kill(proc.pid, signal.SIGTERM)
        self.assertTrue(proc.is_alive())
        proc.join()
        self.assertFalse(proc.is_alive())
        time_after = time.time()
        self.assertTrue(time_after - time_before > graceful_shutdown_timeout)