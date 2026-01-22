import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
class TestPollGreen(GreenTest, TestPoll):
    Poller = gzmq.Poller

    def test_wakeup(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        poller = self.Poller()
        poller.register(s2, zmq.POLLIN)
        tic = time.perf_counter()
        r = gevent.spawn(lambda: poller.poll(10000))
        s = gevent.spawn(lambda: s1.send(b'msg1'))
        r.join()
        toc = time.perf_counter()
        assert toc - tic < 1

    def test_socket_poll(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        tic = time.perf_counter()
        r = gevent.spawn(lambda: s2.poll(10000))
        s = gevent.spawn(lambda: s1.send(b'msg1'))
        r.join()
        toc = time.perf_counter()
        assert toc - tic < 1