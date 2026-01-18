import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_producer_consumer(self):
    q = queues.Queue(maxsize=3)
    history = []

    @gen.coroutine
    def consumer():
        while True:
            history.append((yield q.get()))
            q.task_done()

    @gen.coroutine
    def producer():
        for item in range(10):
            yield q.put(item)
    consumer()
    yield producer()
    yield q.join()
    self.assertEqual(list(range(10)), history)