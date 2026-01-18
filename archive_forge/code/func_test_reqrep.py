import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
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