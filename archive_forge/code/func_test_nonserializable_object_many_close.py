import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
def test_nonserializable_object_many_close(self):
    self.q.push(b'a', 3)
    self.q.push(b'b', 1)
    self.assertRaises(TypeError, self.q.push, lambda x: x, 0)
    self.q.push(b'c', 2)
    self.assertEqual(self.q.pop(), b'b')
    self.assertEqual(sorted(self.q.close()), [2, 3])