import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
@mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
def test_timeout_task_without_return(self, sleep_mock):
    self.num_runs = 1
    timer = loopingcall.DynamicLoopingCall(self._timeout_task_without_return_but_with_done)
    timer.start(periodic_interval_max=5).wait()
    sleep_mock.assert_has_calls([mock.call(5)])