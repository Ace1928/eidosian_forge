import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_timeout_task_without_return_and_max_periodic(self):
    timer = loopingcall.DynamicLoopingCall(self._timeout_task_without_any_return)
    self.assertRaises(RuntimeError, timer.start().wait)