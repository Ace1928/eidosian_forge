import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def test_task_done_underflow(self):
    q = self.queue_class()
    self.assertRaises(ValueError, q.task_done)