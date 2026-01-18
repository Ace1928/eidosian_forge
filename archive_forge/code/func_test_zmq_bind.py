import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_bind(self):
    ctx = C.zmq_ctx_new()
    socket = C.zmq_socket(ctx, 8)
    assert 0 == C.zmq_bind(socket, b'tcp://*:4444')
    assert ctx != ffi.NULL
    assert ffi.NULL != socket
    assert 0 == C.zmq_close(socket)
    assert 0 == C.zmq_ctx_destroy(ctx)