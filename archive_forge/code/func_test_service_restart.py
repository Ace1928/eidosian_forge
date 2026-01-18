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
def test_service_restart(self):
    ready = self._spawn()
    timeout = 5
    ready.wait(timeout)
    self.assertTrue(ready.is_set(), 'Service never became ready')
    ready.clear()
    os.kill(self.pid, signal.SIGHUP)
    ready.wait(timeout)
    self.assertTrue(ready.is_set(), 'Service never back after SIGHUP')