import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
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