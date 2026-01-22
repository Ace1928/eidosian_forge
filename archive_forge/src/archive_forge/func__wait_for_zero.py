import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
def _wait_for_zero(self):
    """Called at an interval until num_runs == 0."""
    if self.num_runs == 0:
        raise loopingcall.LoopingCallDone(False)
    else:
        self.num_runs = self.num_runs - 1
        sleep_for = self.num_runs * 10 + 1
        return sleep_for