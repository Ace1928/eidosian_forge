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
def test_shuts_down_on_sigint_when_client_connected(self):
    proc, conn = self.run_server()
    self.assertTrue(proc.is_alive())
    os.kill(proc.pid, signal.SIGINT)
    proc.join()
    conn.close()