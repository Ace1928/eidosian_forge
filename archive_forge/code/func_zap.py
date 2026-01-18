import contextlib
import os
import time
from threading import Thread
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
from zmq.utils import z85
@contextlib.contextmanager
def zap(self):
    self.start_zap()
    time.sleep(0.5)
    try:
        yield
    finally:
        self.stop_zap()