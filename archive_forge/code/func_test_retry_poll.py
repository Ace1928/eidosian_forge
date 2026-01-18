import signal
import time
from threading import Thread
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, SkipTest
@mark.flaky(reruns=3)
def test_retry_poll(self):
    x, y = self.create_bound_pair()
    poller = zmq.Poller()
    poller.register(x, zmq.POLLIN)
    self.alarm()

    def send():
        time.sleep(2 * self.signal_delay)
        y.send(b'ping')
    t = Thread(target=send)
    t.start()
    evts = dict(poller.poll(2 * self.timeout_ms))
    t.join()
    assert x in evts
    assert self.timer_fired
    x.recv()