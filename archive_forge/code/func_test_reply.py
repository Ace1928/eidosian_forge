import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def test_reply(self):
    alice, bob, mon = self.build_device()
    alices = b'hello bob'.split()
    alice.send_multipart(alices)
    bobs = self.recv_multipart(bob)
    assert alices == bobs
    bobs = b'hello alice'.split()
    bob.send_multipart(bobs)
    alices = self.recv_multipart(alice)
    assert alices == bobs
    self.teardown_device()