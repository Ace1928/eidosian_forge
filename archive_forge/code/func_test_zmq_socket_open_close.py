import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_socket_open_close(self):
    ctx = C.zmq_ctx_new()
    socket = C.zmq_socket(ctx, PUSH)
    assert ctx != ffi.NULL
    assert ffi.NULL != socket
    assert 0 == C.zmq_close(socket)
    assert 0 == C.zmq_ctx_destroy(ctx)