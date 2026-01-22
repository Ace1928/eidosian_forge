import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class LifoQueueJoinTest(QueueJoinTest):
    queue_class = queues.LifoQueue

    @gen_test
    def test_order(self):
        q = self.queue_class(maxsize=2)
        q.put_nowait(1)
        q.put_nowait(0)
        self.assertTrue(q.full())
        q.put(3)
        q.put(2)
        self.assertEqual(3, q.get_nowait())
        self.assertEqual(2, (yield q.get()))
        self.assertEqual(0, q.get_nowait())
        self.assertEqual(1, (yield q.get()))
        self.assertTrue(q.empty())