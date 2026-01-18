import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_setsockopt(self):
    ctx = C.zmq_ctx_new()
    socket = C.zmq_socket(ctx, PUSH)
    identity = ffi.new('char[3]', b'zmq')
    ret = C.zmq_setsockopt(socket, IDENTITY, ffi.cast('void*', identity), 3)
    assert ret == 0
    assert ctx != ffi.NULL
    assert ffi.NULL != socket
    assert 0 == C.zmq_close(socket)
    assert 0 == C.zmq_ctx_destroy(ctx)