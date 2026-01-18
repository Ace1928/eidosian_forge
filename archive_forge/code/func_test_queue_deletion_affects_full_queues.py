import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def test_queue_deletion_affects_full_queues(self):
    conn = SQSConnection()
    initial_count = len(conn.get_all_queues())
    empty = conn.create_queue('empty%d' % int(time.time()))
    full = conn.create_queue('full%d' % int(time.time()))
    time.sleep(60)
    self.assertEqual(len(conn.get_all_queues()), initial_count + 2)
    m1 = Message()
    m1.set_body('This is a test message.')
    full.write(m1)
    self.assertEqual(full.count(), 1)
    self.assertTrue(conn.delete_queue(empty))
    self.assertTrue(conn.delete_queue(full))
    time.sleep(90)
    self.assertEqual(len(conn.get_all_queues()), initial_count)