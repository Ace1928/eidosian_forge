import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
def test_pair(self):
    s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    wait()
    rlist, wlist, xlist = zmq.select([s1, s2], [s1, s2], [s1, s2])
    assert s1 in wlist
    assert s2 in wlist
    assert s1 not in rlist
    assert s2 not in rlist