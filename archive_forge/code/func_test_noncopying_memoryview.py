import copy
import gc
import sys
import time
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest, skip_pypy
def test_noncopying_memoryview(self):
    """test non-copying memmoryview messages"""
    null = b'\x00' * 64
    sa, sb = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
    for i in range(32):
        sb.send(memoryview(null), copy=False)
        m = sa.recv(copy=False)
        buf = memoryview(m)
        for i in range(5):
            ff = b'\xff' * (40 + i * 10)
            sb.send(memoryview(ff), copy=False)
            m2 = sa.recv(copy=False)
            buf2 = memoryview(m2)
            assert buf.tobytes() == null
            assert not buf.readonly
            assert buf2.tobytes() == ff
            assert not buf2.readonly
            assert type(buf) is memoryview