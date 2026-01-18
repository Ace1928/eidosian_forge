import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_terminate_on_exception(self):

    def _raise_it():
        raise RuntimeError()
    timer = loopingcall.DynamicLoopingCall(_raise_it)
    self.assertRaises(RuntimeError, timer.start().wait)