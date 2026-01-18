import struct
import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, SkipTest
def test_proxy_steerable_bind_to_random_with_args(self):
    if zmq.zmq_version_info() < (4, 1):
        raise SkipTest('Steerable Proxies only in libzmq >= 4.1')
    dev = devices.ThreadProxySteerable(zmq.PULL, zmq.PUSH, zmq.PUSH, zmq.PAIR)
    iface = 'tcp://127.0.0.1'
    ports = []
    min, max = (5000, 5050)
    ports.extend([dev.bind_in_to_random_port(iface, min_port=min, max_port=max), dev.bind_out_to_random_port(iface, min_port=min, max_port=max), dev.bind_mon_to_random_port(iface, min_port=min, max_port=max), dev.bind_ctrl_to_random_port(iface, min_port=min, max_port=max)])
    for port in ports:
        if port < min or port > max:
            self.fail('Unexpected port number: %i' % port)