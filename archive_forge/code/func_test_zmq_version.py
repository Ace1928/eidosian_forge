from unittest import TestCase
import zmq
from zmq.sugar import version
def test_zmq_version(self):
    v = zmq.zmq_version()
    assert isinstance(v, str)