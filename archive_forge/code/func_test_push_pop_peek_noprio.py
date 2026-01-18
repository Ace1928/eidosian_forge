import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_push_pop_peek_noprio(self):
    self.assertEqual(self.q.peek(), None)
    self.q.push(b'a')
    self.q.push(b'b')
    self.q.push(b'c')
    self.assertEqual(self.q.peek(), b'c')
    self.assertEqual(self.q.pop(), b'c')
    self.assertEqual(self.q.peek(), b'b')
    self.assertEqual(self.q.pop(), b'b')
    self.assertEqual(self.q.peek(), b'a')
    self.assertEqual(self.q.pop(), b'a')
    self.assertEqual(self.q.peek(), None)
    self.assertEqual(self.q.pop(), None)