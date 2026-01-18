import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_notify_all_with_timeout(self):
    c = locks.Condition()
    self.record_done(c.wait(), 0)
    self.record_done(c.wait(timedelta(seconds=0.01)), 1)
    self.record_done(c.wait(), 2)
    yield gen.sleep(0.02)
    self.assertEqual(['timeout'], self.history)
    c.notify_all()
    yield
    self.assertEqual(['timeout', 0, 2], self.history)