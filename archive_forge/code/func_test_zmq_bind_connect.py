import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_bind_connect(self):
    ctx = C.zmq_ctx_new()
    socket1 = C.zmq_socket(ctx, PUSH)
    socket2 = C.zmq_socket(ctx, PULL)
    assert 0 == C.zmq_bind(socket1, b'tcp://*:4444')
    assert 0 == C.zmq_connect(socket2, b'tcp://127.0.0.1:4444')
    assert ctx != ffi.NULL
    assert ffi.NULL != socket1
    assert ffi.NULL != socket2
    assert 0 == C.zmq_close(socket1)
    assert 0 == C.zmq_close(socket2)
    assert 0 == C.zmq_ctx_destroy(ctx)