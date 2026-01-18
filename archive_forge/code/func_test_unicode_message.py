import logging
import time
import zmq
from zmq.log import handlers
from zmq.tests import BaseZMQTestCase
def test_unicode_message(self):
    logger, handler, sub = self.connect_handler()
    base_topic = (self.topic + '.INFO').encode()
    for msg, expected in [('hello', [base_topic, b'hello\n']), ('héllo', [base_topic, 'héllo\n'.encode()]), ('tøpic::héllo', [base_topic + '.tøpic'.encode(), 'héllo\n'.encode()])]:
        logger.info(msg)
        received = sub.recv_multipart()
        assert received == expected
    logger.removeHandler(handler)