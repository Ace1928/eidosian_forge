import time
from unittest import TestCase
from zmq.tests import SkipTest
def test_zmq_recv(self):
    ctx = C.zmq_ctx_new()
    sender = C.zmq_socket(ctx, REQ)
    receiver = C.zmq_socket(ctx, REP)
    assert 0 == C.zmq_bind(receiver, b'tcp://*:2222')
    assert 0 == C.zmq_connect(sender, b'tcp://127.0.0.1:2222')
    time.sleep(0.1)
    zmq_msg = ffi.new('zmq_msg_t*')
    message = ffi.new('char[5]', b'Hello')
    C.zmq_msg_init_data(zmq_msg, ffi.cast('void*', message), ffi.cast('size_t', 5), ffi.NULL, ffi.NULL)
    zmq_msg2 = ffi.new('zmq_msg_t*')
    C.zmq_msg_init(zmq_msg2)
    assert 5 == C.zmq_msg_send(zmq_msg, sender, 0)
    assert 5 == C.zmq_msg_recv(zmq_msg2, receiver, 0)
    assert 5 == C.zmq_msg_size(zmq_msg2)
    assert b'Hello' == ffi.buffer(C.zmq_msg_data(zmq_msg2), C.zmq_msg_size(zmq_msg2))[:]
    assert C.zmq_close(sender) == 0
    assert C.zmq_close(receiver) == 0
    assert C.zmq_ctx_destroy(ctx) == 0