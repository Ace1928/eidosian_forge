import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_notify_1(self):
    c = locks.Condition()
    self.record_done(c.wait(), 'wait1')
    self.record_done(c.wait(), 'wait2')
    c.notify(1)
    self.loop_briefly()
    self.history.append('notify1')
    c.notify(1)
    self.loop_briefly()
    self.history.append('notify2')
    self.assertEqual(['wait1', 'notify1', 'wait2', 'notify2'], self.history)