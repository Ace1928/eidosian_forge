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
def test_parent_signal_sighup(self):
    start_workers = self._spawn()
    os.kill(self.pid, signal.SIGHUP)

    def cond():
        workers = self._get_workers()
        return len(workers) == len(start_workers) and (not set(start_workers).intersection(workers))
    timeout = 10
    self._wait(cond, timeout)
    self.assertTrue(cond())