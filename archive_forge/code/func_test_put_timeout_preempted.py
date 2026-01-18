import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_put_timeout_preempted(self):
    q = queues.Queue(1)
    q.put_nowait(0)
    put = q.put(1, timeout=timedelta(seconds=0.01))
    q.get()
    yield gen.sleep(0.02)
    yield put