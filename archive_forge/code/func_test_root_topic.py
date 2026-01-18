import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_root_topic(self):
    logger, handler, sub = self.connect_handler()
    handler.socket.bind(self.iface)
    sub2 = sub.context.socket(zmq.SUB)
    self.sockets.append(sub2)
    sub2.connect(self.iface)
    sub2.setsockopt(zmq.SUBSCRIBE, b'')
    handler.root_topic = b'twoonly'
    msg1 = 'ignored'
    logger.info(msg1)
    self.assertRaisesErrno(zmq.EAGAIN, sub.recv, zmq.NOBLOCK)
    topic, msg2 = sub2.recv_multipart()
    assert topic == b'twoonly.INFO'
    assert msg2 == (msg1 + '\n').encode()
    logger.removeHandler(handler)