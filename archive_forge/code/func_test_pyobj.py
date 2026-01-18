import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent
def test_pyobj(self):
    s1, s2 = self.create_bound_pair(zmq.REQ, zmq.REP)
    o = dict(a=10, b=range(10))
    self.ping_pong_pyobj(s1, s2, o)