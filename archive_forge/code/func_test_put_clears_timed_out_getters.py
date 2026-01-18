import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_put_clears_timed_out_getters(self):
    q = queues.Queue()
    getters = [asyncio.ensure_future(q.get(timedelta(seconds=0.01))) for _ in range(10)]
    get = asyncio.ensure_future(q.get())
    q.get()
    self.assertEqual(12, len(q._getters))
    yield gen.sleep(0.02)
    self.assertEqual(12, len(q._getters))
    self.assertFalse(get.done())
    q.put(0)
    self.assertEqual(1, len(q._getters))
    self.assertEqual(0, (yield get))
    for getter in getters:
        self.assertRaises(TimeoutError, getter.result)