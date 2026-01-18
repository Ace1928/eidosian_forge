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
def test_terminate_sigterm(self):
    ready = self._spawn()
    timeout = 5
    ready.wait(timeout)
    self.assertTrue(ready.is_set(), 'Service never became ready')
    os.kill(self.pid, signal.SIGTERM)
    status = self._reap_test()
    self.assertTrue(os.WIFEXITED(status))
    self.assertEqual(0, os.WEXITSTATUS(status))