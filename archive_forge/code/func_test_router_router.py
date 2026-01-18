import threading
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase
def test_router_router(self):
    """test router-router MQ devices"""
    dev = devices.ThreadMonitoredQueue(zmq.ROUTER, zmq.ROUTER, zmq.PUB, b'in', b'out')
    self.device = dev
    dev.setsockopt_in(zmq.LINGER, 0)
    dev.setsockopt_out(zmq.LINGER, 0)
    dev.setsockopt_mon(zmq.LINGER, 0)
    porta = dev.bind_in_to_random_port('tcp://127.0.0.1')
    portb = dev.bind_out_to_random_port('tcp://127.0.0.1')
    a = self.context.socket(zmq.DEALER)
    a.identity = b'a'
    b = self.context.socket(zmq.DEALER)
    b.identity = b'b'
    self.sockets.extend([a, b])
    a.connect('tcp://127.0.0.1:%i' % porta)
    b.connect('tcp://127.0.0.1:%i' % portb)
    dev.start()
    time.sleep(1)
    if zmq.zmq_version_info() >= (3, 1, 0):
        ping_msg = [b'ping', b'pong']
        for s in (a, b):
            s.send_multipart(ping_msg)
            try:
                s.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
    msg = [b'hello', b'there']
    a.send_multipart([b'b'] + msg)
    bmsg = self.recv_multipart(b)
    assert bmsg == [b'a'] + msg
    b.send_multipart(bmsg)
    amsg = self.recv_multipart(a)
    assert amsg == [b'b'] + msg
    self.teardown_device()