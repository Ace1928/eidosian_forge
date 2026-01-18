from threading import Thread
import zmq
from zmq import Again, ContextTerminated, ZMQError, strerror
from zmq.tests import BaseZMQTestCase
def test_zmqerror(self):
    for errno in range(10):
        e = ZMQError(errno)
        assert e.errno == errno
        assert str(e) == strerror(errno)