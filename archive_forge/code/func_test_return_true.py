import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def test_return_true(self):

    def _raise_it():
        raise loopingcall.LoopingCallDone(True)
    timer = loopingcall.DynamicLoopingCall(_raise_it)
    self.assertTrue(timer.start().wait())