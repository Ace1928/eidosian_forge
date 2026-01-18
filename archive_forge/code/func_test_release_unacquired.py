import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_release_unacquired(self):
    sem = locks.BoundedSemaphore()
    self.assertRaises(ValueError, sem.release)
    sem.acquire()
    future = asyncio.ensure_future(sem.acquire())
    self.assertFalse(future.done())
    sem.release()
    self.assertTrue(future.done())
    sem.release()
    self.assertRaises(ValueError, sem.release)