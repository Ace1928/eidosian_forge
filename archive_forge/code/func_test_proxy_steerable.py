import struct
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
def test_proxy_steerable(self):
    if zmq.zmq_version_info() < (4, 1):
        raise SkipTest('Steerable Proxies only in libzmq >= 4.1')
    if zmq.zmq_version_info() >= (4, 3, 5):
        raise SkipTest('Steerable Proxies removed in libzmq 4.3.5')
    dev = devices.ThreadProxySteerable(zmq.PULL, zmq.PUSH, zmq.PUSH, zmq.PAIR)
    iface = 'tcp://127.0.0.1'
    port = dev.bind_in_to_random_port(iface)
    port2 = dev.bind_out_to_random_port(iface)
    port3 = dev.bind_mon_to_random_port(iface)
    port4 = dev.bind_ctrl_to_random_port(iface)
    dev.start()
    time.sleep(0.25)
    msg = b'hello'
    push = self.context.socket(zmq.PUSH)
    push.connect('%s:%i' % (iface, port))
    pull = self.context.socket(zmq.PULL)
    pull.connect('%s:%i' % (iface, port2))
    mon = self.context.socket(zmq.PULL)
    mon.connect('%s:%i' % (iface, port3))
    ctrl = self.context.socket(zmq.PAIR)
    ctrl.connect('%s:%i' % (iface, port4))
    push.send(msg)
    self.sockets.extend([push, pull, mon, ctrl])
    assert msg == self.recv(pull)
    assert msg == self.recv(mon)
    ctrl.send(b'TERMINATE')
    dev.join()