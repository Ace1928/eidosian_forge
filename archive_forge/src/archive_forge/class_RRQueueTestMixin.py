import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class RRQueueTestMixin:

    def setUp(self):
        super().setUp()
        self.q = RoundRobinQueue(self.qfactory)

    def qfactory(self, key):
        raise NotImplementedError

    def test_len_nonzero(self):
        assert not self.q
        self.assertEqual(len(self.q), 0)
        self.q.push(b'a', '3')
        assert self.q
        self.q.push(b'b', '1')
        self.q.push(b'c', '2')
        self.q.push(b'd', '1')
        self.assertEqual(len(self.q), 4)
        self.q.pop()
        self.q.pop()
        self.q.pop()
        self.q.pop()
        assert not self.q
        self.assertEqual(len(self.q), 0)

    def test_close(self):
        self.q.push(b'a', '3')
        self.q.push(b'b', '1')
        self.q.push(b'c', '2')
        self.q.push(b'd', '1')
        iqueues = self.q.queues.values()
        self.assertEqual(sorted(self.q.close()), ['1', '2', '3'])
        assert all((q.closed for q in iqueues))

    def test_close_return_active(self):
        self.q.push(b'b', '1')
        self.q.push(b'c', '2')
        self.q.push(b'a', '3')
        self.q.pop()
        self.assertEqual(sorted(self.q.close()), ['2', '3'])