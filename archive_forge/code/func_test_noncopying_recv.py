import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_noncopying_recv(self):
    """check for clobbering message buffers"""
    null = b'\x00' * 64
    sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    for i in range(32):
        sb.send(null, copy=False)
        m = sa.recv(copy=False)
        mb = m.bytes
        buf = m.buffer
        del m
        for i in range(5):
            ff = b'\xff' * (40 + i * 10)
            sb.send(ff, copy=False)
            m2 = sa.recv(copy=False)
            b = buf.tobytes()
            assert b == null
            assert mb == null
            assert m2.bytes == ff
            assert type(m2.bytes) is bytes