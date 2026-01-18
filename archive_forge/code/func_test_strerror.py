from threading import Thread
import zmq
from zmq import Again, ContextTerminated, ZMQError, strerror
from zmq.tests import BaseZMQTestCase
def test_strerror(self):
    """test that strerror gets the right type."""
    for i in range(10):
        e = strerror(i)
        assert isinstance(e, str)