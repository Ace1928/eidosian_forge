import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
@mark.skipif(not hasattr(zmq, 'SNDTIMEO'), reason='requires SNDTIMEO')
def test_retry_send(self):
    push = self.socket(zmq.PUSH)
    push.sndtimeo = self.timeout_ms
    self.alarm()
    self.assertRaises(zmq.Again, push.send, b'buf')
    assert self.timer_fired