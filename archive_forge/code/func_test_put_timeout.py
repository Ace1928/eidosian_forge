import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_put_timeout(self):
    q = queues.Queue(1)
    q.put_nowait(0)
    put_timeout = q.put(1, timeout=timedelta(seconds=0.01))
    put = q.put(2)
    with self.assertRaises(TimeoutError):
        yield put_timeout
    self.assertEqual(0, q.get_nowait())
    self.assertEqual(2, (yield q.get()))
    yield put