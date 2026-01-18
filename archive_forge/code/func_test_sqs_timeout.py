import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def test_sqs_timeout(self):
    c = SQSConnection()
    queue_name = 'test_sqs_timeout_%s' % int(time.time())
    queue = c.create_queue(queue_name)
    self.addCleanup(c.delete_queue, queue, True)
    start = time.time()
    poll_seconds = 2
    response = queue.read(visibility_timeout=None, wait_time_seconds=poll_seconds)
    total_time = time.time() - start
    self.assertTrue(total_time > poll_seconds, 'SQS queue did not block for at least %s seconds: %s' % (poll_seconds, total_time))
    self.assertIsNone(response)
    c.send_message(queue, 'test message')
    start = time.time()
    poll_seconds = 2
    message = c.receive_message(queue, number_messages=1, visibility_timeout=None, attributes=None, wait_time_seconds=poll_seconds)[0]
    total_time = time.time() - start
    self.assertTrue(total_time < poll_seconds, 'SQS queue blocked longer than %s seconds: %s' % (poll_seconds, total_time))
    self.assertEqual(message.get_body(), 'test message')
    attrs = c.get_queue_attributes(queue, 'ReceiveMessageWaitTimeSeconds')
    self.assertEqual(attrs['ReceiveMessageWaitTimeSeconds'], '0')