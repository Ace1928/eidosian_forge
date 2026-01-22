import asyncio
from datetime import timedelta
from random import random
import unittest
from tornado import gen, queues
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class QueueBasicTest(AsyncTestCase):

    def test_repr_and_str(self):
        q = queues.Queue(maxsize=1)
        self.assertIn(hex(id(q)), repr(q))
        self.assertNotIn(hex(id(q)), str(q))
        q.get()
        for q_str in (repr(q), str(q)):
            self.assertTrue(q_str.startswith('<Queue'))
            self.assertIn('maxsize=1', q_str)
            self.assertIn('getters[1]', q_str)
            self.assertNotIn('putters', q_str)
            self.assertNotIn('tasks', q_str)
        q.put(None)
        q.put(None)
        q.put(None)
        for q_str in (repr(q), str(q)):
            self.assertNotIn('getters', q_str)
            self.assertIn('putters[1]', q_str)
            self.assertIn('tasks=2', q_str)

    def test_order(self):
        q = queues.Queue()
        for i in [1, 3, 2]:
            q.put_nowait(i)
        items = [q.get_nowait() for _ in range(3)]
        self.assertEqual([1, 3, 2], items)

    @gen_test
    def test_maxsize(self):
        self.assertRaises(TypeError, queues.Queue, maxsize=None)
        self.assertRaises(ValueError, queues.Queue, maxsize=-1)
        q = queues.Queue(maxsize=2)
        self.assertTrue(q.empty())
        self.assertFalse(q.full())
        self.assertEqual(2, q.maxsize)
        self.assertTrue(q.put(0).done())
        self.assertTrue(q.put(1).done())
        self.assertFalse(q.empty())
        self.assertTrue(q.full())
        put2 = q.put(2)
        self.assertFalse(put2.done())
        self.assertEqual(0, (yield q.get()))
        self.assertTrue(put2.done())
        self.assertFalse(q.empty())
        self.assertTrue(q.full())