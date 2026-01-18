import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_set_info_formatter_via_property(self):
    logger, handler, sub = self.connect_handler()
    handler.formatters[logging.INFO] = logging.Formatter('%(message)s UNITTEST\n')
    handler.socket.bind(self.iface)
    sub.setsockopt(zmq.SUBSCRIBE, handler.root_topic.encode())
    logger.info('info message')
    topic, msg = sub.recv_multipart()
    assert msg == b'info message UNITTEST\n'
    logger.removeHandler(handler)