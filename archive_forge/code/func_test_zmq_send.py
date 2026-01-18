import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_send(self):
    ctx = C.zmq_ctx_new()
    sender = C.zmq_socket(ctx, REQ)
    receiver = C.zmq_socket(ctx, REP)
    assert 0 == C.zmq_bind(receiver, b'tcp://*:7777')
    assert 0 == C.zmq_connect(sender, b'tcp://127.0.0.1:7777')
    time.sleep(0.1)
    zmq_msg = ffi.new('zmq_msg_t*')
    message = ffi.new('char[5]', b'Hello')
    C.zmq_msg_init_data(zmq_msg, ffi.cast('void*', message), ffi.cast('size_t', 5), ffi.NULL, ffi.NULL)
    assert 5 == C.zmq_msg_send(zmq_msg, sender, 0)
    assert 0 == C.zmq_msg_close(zmq_msg)
    assert C.zmq_close(sender) == 0
    assert C.zmq_close(receiver) == 0
    assert C.zmq_ctx_destroy(ctx) == 0