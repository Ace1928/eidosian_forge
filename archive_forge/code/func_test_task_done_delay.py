import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_task_done_delay(self):
    q = self.queue_class()
    q.put_nowait(0)
    join = asyncio.ensure_future(q.join())
    self.assertFalse(join.done())
    yield q.get()
    self.assertFalse(join.done())
    yield gen.moment
    self.assertFalse(join.done())
    q.task_done()
    self.assertTrue(join.done())