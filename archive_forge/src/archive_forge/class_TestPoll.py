import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
class TestPoll(PollZMQTestCase):
    Poller = zmq.Poller

    def test_pair(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        wait()
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN | zmq.POLLOUT)
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT
        assert socks[s2] == zmq.POLLOUT
        s1.send(b'msg1')
        s2.send(b'msg2')
        wait()
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT | zmq.POLLIN
        assert socks[s2] == zmq.POLLOUT | zmq.POLLIN
        s1.recv()
        s2.recv()
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT
        assert socks[s2] == zmq.POLLOUT
        poller.unregister(s1)
        poller.unregister(s2)

    def test_reqrep(self):
        s1, s2 = self.create_bound_pair(zmq.REP, zmq.REQ)
        wait()
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN | zmq.POLLOUT)
        socks = dict(poller.poll())
        assert s1 not in socks
        assert socks[s2] == zmq.POLLOUT
        s2.send(b'msg1')
        socks = dict(poller.poll())
        assert s2 not in socks
        time.sleep(0.5)
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLIN
        s1.recv()
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT
        s1.send(b'msg2')
        socks = dict(poller.poll())
        assert s1 not in socks
        time.sleep(0.5)
        socks = dict(poller.poll())
        assert socks[s2] == zmq.POLLIN
        s2.recv()
        socks = dict(poller.poll())
        assert socks[s2] == zmq.POLLOUT
        poller.unregister(s1)
        poller.unregister(s2)

    def test_no_events(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
        poller.register(s2, 0)
        assert s1 in poller
        assert s2 not in poller
        poller.register(s1, 0)
        assert s1 not in poller

    def test_pubsub(self):
        s1, s2 = self.create_bound_pair(zmq.PUB, zmq.SUB)
        s2.setsockopt(zmq.SUBSCRIBE, b'')
        wait()
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN | zmq.POLLOUT)
        poller.register(s2, zmq.POLLIN)
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT
        assert s2 not in socks
        s1.send(b'msg1')
        socks = dict(poller.poll())
        assert socks[s1] == zmq.POLLOUT
        wait()
        socks = dict(poller.poll())
        assert socks[s2] == zmq.POLLIN
        s2.recv()
        socks = dict(poller.poll())
        assert s2 not in socks
        poller.unregister(s1)
        poller.unregister(s2)

    @mark.skipif(sys.platform.startswith('win'), reason='Windows')
    def test_raw(self):
        r, w = os.pipe()
        r = os.fdopen(r, 'rb')
        w = os.fdopen(w, 'wb')
        p = self.Poller()
        p.register(r, zmq.POLLIN)
        socks = dict(p.poll(1))
        assert socks == {}
        w.write(b'x')
        w.flush()
        socks = dict(p.poll(1))
        assert socks == {r.fileno(): zmq.POLLIN}
        w.close()
        r.close()

    @mark.flaky(reruns=3)
    def test_timeout(self):
        """make sure Poller.poll timeout has the right units (milliseconds)."""
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        poller = self.Poller()
        poller.register(s1, zmq.POLLIN)
        tic = time.perf_counter()
        poller.poll(0.005)
        toc = time.perf_counter()
        toc - tic < 0.5
        tic = time.perf_counter()
        poller.poll(50)
        toc = time.perf_counter()
        assert toc - tic < 0.5
        assert toc - tic > 0.01
        tic = time.perf_counter()
        poller.poll(500)
        toc = time.perf_counter()
        assert toc - tic < 1
        assert toc - tic > 0.1