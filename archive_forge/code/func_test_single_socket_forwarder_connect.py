import time
import zmq
from zmq import devices
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest, have_gevent
def test_single_socket_forwarder_connect(self):
    if zmq.zmq_version() in ('4.1.1', '4.0.6'):
        raise SkipTest('libzmq-%s broke single-socket devices' % zmq.zmq_version())
    dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
    req = self.context.socket(zmq.REQ)
    port = req.bind_to_random_port('tcp://127.0.0.1')
    dev.connect_in('tcp://127.0.0.1:%i' % port)
    dev.start()
    time.sleep(0.25)
    msg = b'hello'
    req.send(msg)
    assert msg == self.recv(req)
    del dev
    req.close()
    dev = devices.ThreadDevice(zmq.QUEUE, zmq.REP, -1)
    req = self.context.socket(zmq.REQ)
    port = req.bind_to_random_port('tcp://127.0.0.1')
    dev.connect_out('tcp://127.0.0.1:%i' % port)
    dev.start()
    time.sleep(0.25)
    msg = b'hello again'
    req.send(msg)
    assert msg == self.recv(req)
    del dev
    req.close()