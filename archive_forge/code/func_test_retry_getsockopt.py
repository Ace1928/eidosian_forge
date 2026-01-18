import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
def test_retry_getsockopt(self):
    raise SkipTest('TODO: find a way to interrupt getsockopt')